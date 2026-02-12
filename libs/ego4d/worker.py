"""
for iter snag model
"""
from collections import OrderedDict
from copy import deepcopy
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .dataset import make_dataset, make_dataloader
from ..utils.dist_utils import get_rank, get_world_size, barrier, print0
from .model import (
    PtGenerator,
)
from ..modeling import (
    make_optimizer, make_scheduler
)
from ..utils.worker_utils import (
    calc_focal_loss,
    calc_iou_loss,
    annotate_points_per_video,
)
from ..utils.worker_utils import BaseTrainer, BaseEvaluator
from ..nms import batched_nms
from ..utils.train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str, easy_reduce

from collections import defaultdict
from .model import Ego4dGrounder


STOP_ITR = None

def create_model(opt):
    model = Ego4dGrounder(opt)
    return model

class Trainer(BaseTrainer):

    def __init__(self, opt, wandb_run=None):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        self.model = create_model(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        print0("Building dataset...")
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(opt, num_epochs=self.num_epochs, is_training=True)
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0
        print0('Epoch Size:', len(self.dataloader))

        # build training utilities
        self.itrs_per_epoch = opt['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['scheduler'])
        self.clip_grad_norm = opt['optimizer']['clip_grad_norm']

        # build logging utilities
        self.log_interval = opt.aux.log_interval # ['log'].get('log_interval', 100)
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.logger.write("-------------------------------------------------------------------\n")
            self.logger.write(f'seed {opt.seed}')
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()])
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        if self.opt.aux.eval_run > 0:
            self.evaluator = Evaluator(self.opt, train_time=True)

        self.wandb_run = wandb_run

    def evaluate(self, ct):
        self.checkpoint()
        if get_rank() == 0 and (self.opt.aux.eval_run > 0) and (ct % self.opt.aux.eval_run == 0):
            self.evaluator.run(train_time_data=[self.model_ema, self.wandb_run, self.itr, self.epoch])
            self.evaluator.reset()
        barrier()

    def _microbatch_forward_backward(self, data_list, is_last=False, backward=True):

        model = self.model

        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        sidekick_vid, _ = self._batchify_videos(
            vid_list=[d['sidekick_vid'] for d in data_list], 
            )
        sidekick_vid = sidekick_vid.cuda(non_blocking=True)

        text_cls = [ d['text_cls'] for d in data_list ]
        text_cls = torch.concat(text_cls).cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)

        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks = \
                model(vid, sidekick_vid, vid_masks, text, text_cls, text_masks, text_size)
        else:
            with model.no_sync():
                fpn_logits1, fpn_logits2,fpn_offsets, fpn_masks = \
                    model(vid, sidekick_vid, vid_masks, text, text_cls, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits1 = torch.cat(fpn_logits1, dim=1)   # (bs, p)
        fpn_logits2 = torch.cat(fpn_logits2, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss1 = calc_focal_loss(
            logits=fpn_logits1[fpn_masks], labels=gt_labels[fpn_masks],
            alpha=self.opt.loss.fc_a,
            smoothing=self.opt.loss.fc_s,
        ) / self.loss_norm * get_world_size()
        cls_loss2 = calc_focal_loss(
            logits=fpn_logits2[fpn_masks], labels=gt_labels[fpn_masks],
            alpha=self.opt.loss.fc_a,
            smoothing=self.opt.loss.fc_s,
        ) / self.loss_norm * get_world_size()
        cls_loss = (cls_loss1 + cls_loss2 ) / 2 
        
        ## (3) regression loss on positive points
        reg_loss = calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks],
            reg_loss = self.reg_loss,
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss

        if backward:
            total_loss.backward()

        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

class Evaluator(BaseEvaluator):

    def __init__(self, opt, train_time=False):

        self.opt = opt

        # set random seed
        rng = None

        # prepare dataset
        self.dataset = make_dataset(opt, is_training=False)
        self.dataloader, _ = make_dataloader(
            self.dataset, is_training=False, generator=rng, batch_size=1, num_workers=opt['train']['num_workers'],
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        if not train_time:
            self.model = create_model(opt).cuda()
            self.load_model()
            self.model.eval().requires_grad_(False)
        else:
            self.model = None
        pt_gen = opt.pt_gen.clone()
        pt_gen.max_seq_len = opt.model.vid_net.max_seq_len * 10
        self.pt_gen = PtGenerator(**pt_gen).cuda()
        # build logging utilities
        root = self.opt._root
        os.makedirs(root, exist_ok=True)
        self.logger = Logger(os.path.join(root, f"eval_{opt['_ckpt']}.txt"), dump_to_file=True) if (not train_time) else None

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size') # None
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

        self.time_dict = defaultdict(list)

    def run(self, train_time_data=None):
        if train_time_data is not None:
            model, wandb_run, writer_it, epoch = train_time_data
            self.logger = Logger(os.path.join(self.opt['_root'], f"eval_{epoch}_{writer_it}.txt")) 
            self.model = model
        else:
            epoch = self.opt['_ckpt']
            writer_it = int(epoch.split('-')[1].split('.')[0])
            epoch = int(epoch.split('-')[0])
            print(epoch, writer_it)

        print0("Evaluation started.")
        start_time = time.time()
        if train_time_data:
            loader = self.dataloader
        else:
            loader = tqdm(self.dataloader)
        for i, data_list in enumerate(loader):

            _, results = self.simple_predict(data_list[0])
            targets = data_list[0]['segment']

            assert len(results) == len(targets)
            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = []
                for i in self.ranks:
                    tmp = iou_topk[:i]
                    if len(tmp) > 0:
                        iou_n.append(tmp.max().item())
                    else:
                        iou_n.append(0)
                iou_n = np.array(iou_n)

                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            self.text_cnt += len(targets)

            self.itr += 1

            if self.opt.aux.dryrun:
                break

        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" # if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        wandb_dict = {}
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
                wandb_dict[f"eval/R@{rank}-{thresh:.1f}"] = metrics[i, j]

        self.logger.write(log_str)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")
        if train_time_data and wandb_run is not None:
            wandb_dict[f"eval/epoch"] = int(epoch)
            wandb_run.log(wandb_dict, step=writer_it)

    def _forward(self, data):

        assert self.window_size is None, "sliding-window evaluation is not supported"
        assert self.window_stride is None, "sliding-window evaluation is not supported"

        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        sidekick_vid = data['sidekick_vid']
        vid_len = vid.size(-1)


        # external scores (n, t)
        ext_scores = data['ext_scores'] # None
        if ext_scores is not None and ext_scores.ndim == 1:
            assert False
            ext_scores = ext_scores[None]

        input_vid_len = self.input_vid_len
        if vid_len > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (vid_len + (stride - 1)) // stride * stride

        window = vid
        sidekick_window = sidekick_vid
        window_offset = 0
        window_ext = ext_scores

        window = F.pad(window, (0, input_vid_len - vid_len))[None]
        sidekick_window = F.pad(sidekick_window, (0, input_vid_len - vid_len))[None]
        window_mask = torch.arange(input_vid_len).view(1, -1) < vid_len
        window = window.cuda(non_blocking=True)
        sidekick_window = sidekick_window.cuda(non_blocking=True)
        window_mask = window_mask.cuda(non_blocking=True)
        if window_ext is not None:
            window_ext = F.pad(window_ext, (0, input_vid_len - vid_len))
            window_ext = window_ext.cuda(non_blocking=True)

        text_cls = data['text_cls'].cuda(non_blocking=True)

        out = self.model(window, sidekick_window, window_mask, text_list, text_cls, text_mask_list, evaluate=True)
        fpn_logits_list, fpn_offsets_list, fpn_masks_list = out
        fpn_n_points = [m.size(-1) for m in fpn_masks_list[0]]
        fpn_points = self.pt_gen(fpn_n_points)

        fpn_masks_list = [[m.squeeze(1) for m in fpn_masks] for fpn_masks in fpn_masks_list]

        self.outputs = [fpn_logits_list, fpn_offsets_list, fpn_points, fpn_masks_list]
        return self.outputs

    def _eval_one_video(self, results, targets):
        ct, txt_ct = np.zeros_like(self.counts), np.zeros_like(self.text_cnt)
        iou_list = []
        for result, target in zip(results, targets):
            segs, scores = result['segments'], result['scores']
            idx = scores.argsort(descending=True)
            segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
            target = torch.as_tensor(target, dtype=torch.float)
            target = target.expand(len(segs), -1)

            iou_topk = iou(segs, target)
            iou_n = []
            for i in self.ranks:
                tmp = iou_topk[:i]
                if len(tmp) > 0:
                    iou_n.append(tmp.max().item())
                else:
                    iou_n.append(0)
            iou_n = np.array(iou_n)

            self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            iou_list.append(iou_n)
        self.text_cnt += len(targets)
        return iou_list
