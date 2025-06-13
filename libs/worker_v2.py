"""
for iter snag model
"""
from collections import OrderedDict
from copy import deepcopy
import os
import shutil
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
# from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, PtTransformer,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str
from .helper.utils import easy_reduce
from .helper import utils
# from .helper.azure import easy_send, get_path, log_aml_val

from collections import defaultdict

# def batchify_mv_and_cuda(data_list, mode='raw', input_vid_len=None):
#     if 'mv_data' not in data_list[0]:
#         return None
#     if data_list[0]['mv_data'] is None:
#         return None

#     if mode == 'raw':
#         mv_data = [ d['mv_data'] for d in data_list ]
#         mv = torch.stack([ d[0] for d in mv_data ], dim=0) #.cuda(non_blocking=True)
#         ptype = torch.stack([ d[1] for d in mv_data ], dim=0).cuda(non_blocking=True)
#         f2t_map = torch.stack([ d[2] for d in mv_data ], dim=0).cuda(non_blocking=True)
#         return mv, ptype, f2t_map
#     elif mode == 'feature':
#         mv_data = [ d['mv_data'] for d in data_list ]
#         L = len(mv_data[0])

#         ###############
#         # mv_data = [ torch.stack([ d[i] for d in mv_data ], dim=0).cuda(non_blocking=True) 
#         #                     for i in range(L) ]

#         # for i, d in enumerate(mv_data):
#         #     pad = [0] * (2*len(d.shape))
#         #     pad[3] = input_vid_len - d.size(1)
#         #     pad = pad[::-1]
#         #     import ipdb; ipdb.set_trace()
#         #     d = torch.nn.functional.pad(d, pad, value=0)
#         #     mv_data[i] = d
#         #################

#         new_mv_data = []
#         for l in range(L):
#             dlist = []
#             for d in mv_data: 
#                 d = d[l]
#                 pad = [0] * (2*len(d.shape))
#                 pad[0] = input_vid_len - d.size(0)
#                 pad = pad[::-1]
#                 d = torch.nn.functional.pad(d, pad, value=0)
#                 dlist.append(d)
#             dlist = torch.stack(dlist, dim=0).cuda(non_blocking=True)
#             new_mv_data.append(dlist)
        
#         mv_data = new_mv_data

#         mask = None # no needed for now
#         return mv_data



STOP_ITR = None
# STOP_ITR = 100

def calc_focal_loss(logits, labels, smoothing=0.2, alpha=0.5, reduction='sum'):
    labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
    return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction=reduction)

def calc_iou_loss(pred_offsets, gt_offsets, reg_loss='diou', reduction='sum'):
    iou_loss = ctr_diou_loss if reg_loss == 'diou' else ctr_giou_loss
    return iou_loss(pred_offsets, gt_offsets, reduction=reduction)

def annotate_points_per_video(points, target, center_sampling='radius', center_sampling_radius=1.5):
    """
    Args:
        points (float tensor, (p, 4)): candidate points from all levels.
            (coordinate (1), regression range (2), stride (1))
        target (float tensor, (2,)): ground-truth segment.

    Returns:
        labels (bool tensor, (p,)): ground-truth binary labels.
        offsets (float tensor, (p, 2)): ground-truth offsets.
    """
    # point distance to segment boundaries
    pt2start = points[:, 0] - target[0]     # (p,)
    pt2end = target[1] - points[:, 0]       # (p,)

    # offsets rescaled by down-sampling stride
    offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

    # (1) whether a point lies in given sampling window
    if center_sampling == 'radius':
        ctr = 0.5 * (target[0] + target[1])
        radius = points[:, 3] * center_sampling_radius
        t_min = (ctr - radius).clamp_(min=target[0])
        t_max = (ctr + radius).clamp_(max=target[1])
        # point distance to window boundaries
        pt2left = points[:, 0] - t_min  # (p,)
        pt2right = t_max - points[:, 0] # (p,)
        inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
    else:
        inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

    # (2) whether event is within regression range of a point
    max_reg_dist = torch.maximum(pt2start, pt2end)
    inside_range = torch.logical_and(
        max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
    )

    # a point is positive only if it meets both criteria
    labels = torch.logical_and(inside_window, inside_range)

    return labels, offsets, [inside_window, inside_range]

# def create_train_data(opt, num_epochs=None, is_training=True):
#     rng = fix_random_seed(opt.get('seed', 2022))
#     if num_epochs is None:
#         num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
#     dataset = make_dataset(
#         opt['train']['data'], num_epochs=num_epochs, is_training=True
#     )
#     batch_size = opt['train']['batch_size']
#     dataloader, sampler = make_dataloader(
#         dataset, generator=rng, is_training=True,
#         batch_size=batch_size, num_workers=opt['train']['num_workers'],
#         world_size=get_world_size(), rank=get_rank()
#     )
#     return dataset, dataloader

def create_test_data(opt, split=None):
    rng = fix_random_seed(opt.get('seed', 2022))

    if split is not None:
        opt = deepcopy(opt)
        opt['eval']['data']['split'] = split

    # prepare dataset
    dataset = make_dataset(opt['eval']['data'], is_training=False)
    dataloader, _ = make_dataloader(
        dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
    )
    return dataset, dataloader


# def check_files(dataset, wait=False, ftype='npy'):
#     while True:
#         good = True
#         for vid_id, _ in dataset.vid_dict.items():
#             vid_feat_files = [os.path.join(d, f"{vid_id}.{ftype}") for d in dataset.vid_feat_dir]
#             for f in vid_feat_files:
#                 if not os.path.exists(f):
#                     good = False
        
#         if good:
#             return
#         elif not wait:
#             raise FileNotFoundError()
#         else:
#             print0("Waiting for files...")
#             time.sleep(600)

def create_model(opt):
    # if opt.model.name == 'default':
    #     model = PtTransformer(opt)
    # if opt.model.name == 'early':
    #     from .modeling.model import PtTransformerEarlyFusion
    #     model = PtTransformerEarlyFusion(opt, second_fusion=False)
    # if opt.model.name == 'early2':
    #     from .modeling.model import PtTransformerEarlyFusion
    #     model = PtTransformerEarlyFusion(opt, second_fusion=True)
    if opt.model.name == 'iter':
        from .modeling.model import PtTransformerEarlyFusionIterative
        model = PtTransformerEarlyFusionIterative(opt, second_fusion=False)
    # if opt.model.name == 'iter2':
    #     from .modeling.model import PtTransformerEarlyFusionIterative
    #     model = PtTransformerEarlyFusionIterative(opt, second_fusion=True)

    # elif opt.model.name == 'parallel':
    #     from .modeling.model_new import PtTransformerParallel
    #     model = PtTransformerParallel(opt)
    # elif opt.model.name == 'atten':
    #     from .modeling.model_new import PtTransformerCAttn
    #     model = PtTransformerCAttn(opt)
    # elif opt.model.name == 'var1':
    #     from .modeling.model_new import PtTransformerVariant1
    #     model = PtTransformerVariant1(opt)
    # elif opt.model.name == 'var2':
    #     from .modeling.model_new import PtTransformerVariant2
    #     model = PtTransformerVariant2(opt)
    
    return model

class Trainer:

    def __init__(self, opt, wandb_run=None):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        # self.model = PtTransformer(opt).cuda()
        self.model = create_model(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        if opt.model.pretrain:
            print('Loading pretrained model...', opt.model.pretrain)
            model_ckpt = torch.load(opt.model.pretrain, map_location='cpu')
            self.model.load_state_dict(model_ckpt['model'])
            self.model_ema.load_state_dict(model_ckpt['model_ema'])


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
        # self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.logger.write("-------------------------------------------------------------------\n")
            self.logger.write(f'seed {opt.seed}')
            # self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
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

        # ftype = 'npy' if opt.data.vid_load == 'npy' else 'pk'
        # check_files(self.dataset, wait=True, ftype=ftype)

        if self.opt.aux.eval_run > 0:
            self.evaluator = Evaluator(self.opt, train_time=True)
            # check_files(self.evaluator.dataset, wait=True, ftype=ftype)

        self.wandb_run = wandb_run

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)

            start_time = time.time()

            for data_list in self.dataloader:
                # run one optimization step
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach().item())
                    self.timer.update(time.time() - start_time)
                    start_time = time.time()
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()

                if self.opt.aux.dryrun:
                    break

                if self.opt.aux.eval_by == 'itr' and (self.itr % self.opt.aux.eval_run == 0):
                    self.evaluate(self.itr)
            
                

            self.epoch += 1
            if self.opt.aux.eval_by == 'epoch':
                self.evaluate(self.epoch)

            # self.checkpoint()
            # if get_rank() == 0 and (self.opt.aux.eval_run > 0) and (self.epoch % self.opt.aux.eval_run == 0):
            #     self.evaluator.run(train_time_data=[self.model_ema, self.wandb_run, self.itr, self.epoch])
            #     self.evaluator.reset()
            # barrier()

            if self.opt.aux.dryrun:
                break
        print0("Training completed.")

    def evaluate(self, ct):
        self.checkpoint()
        if get_rank() == 0 and (self.opt.aux.eval_run > 0) and (ct % self.opt.aux.eval_run == 0):
            self.evaluator.run(train_time_data=[self.model_ema, self.wandb_run, self.itr, self.epoch])
            self.evaluator.reset()
        barrier()

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = self.loss_norm_momentum * self.loss_norm \
                        + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)

        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False, backward=True):

        # model = self.model if not use_ema else self.model_ema
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

        shallow_vid, shallow_vid_masks = self._batchify_videos(
            vid_list=[d['shallow_vid'] for d in data_list], 
            )
        shallow_vid = shallow_vid.cuda(non_blocking=True)

        text_cls = [ d['text_cls'] for d in data_list ]
        text_cls = torch.concat(text_cls).cuda(non_blocking=True)

        # if 'mv_data' in data_list[0]:
        #     mv_data = self._batchify_mv_and_cuda(data_list)
        # else:
        #     mv_data = None
        # mv_data = batchify_mv_and_cuda(data_list, mode='feature', input_vid_len=self.input_vid_len)
        mv_data = None

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks = \
                model(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data)
        else:
            with model.no_sync():
                fpn_logits1, fpn_logits2,fpn_offsets, fpn_masks = \
                    model(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data)
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

        # torch.cuda.synchronize()

        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

    # def _batchify_mv_and_cuda(self, data_list):
    #     mv_data = [ d['mv_data'] for d in data_list ]
    #     mv = torch.stack([ d[0] for d in mv_data ], dim=0) #.cuda(non_blocking=True)
    #     ptype = torch.stack([ d[1] for d in mv_data ], dim=0).cuda(non_blocking=True)
    #     f2t_map = torch.stack([ d[2] for d in mv_data ], dim=0).cuda(non_blocking=True)
    #     return mv, ptype, f2t_map


    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = [], []
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list.append(labels)
            offsets_list.append(offsets)
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    # def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
    #     labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
    #     return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    # def _calc_iou_loss(self, pred_offsets, gt_offsets):
    #     iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
    #     return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self, ckpt='last.pth'):
        model_path = os.path.join(self.opt['_root'], 'models', ckpt)
        state_path = os.path.join(self.opt['_root'], 'states', ckpt)
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])

        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(model_ckpt, os.path.join(model_dir, f"{self.epoch}-{self.itr}.pth"))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        # shutil.copyfile(
        #     os.path.join(model_dir, 'last.pth'),
        #     os.path.join(model_dir, f"{self.epoch}-{self.itr}.pth")
        # )

        # if self.opt.aux.is_submit:
        # print0('Sending model checkpoints')
        # if get_rank() == 0:
        # easy_send(os.path.join(state_dir, 'last.pth'))
        # easy_send(os.path.join(model_dir, 'last.pth'))
        # easy_send(os.path.join(model_dir, f"{self.epoch}-{self.itr}.pth"))

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        wandb_dict = {}
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            # log_aml_val('train', k, self.itr, v.item(), double=False)
            wandb_dict['train/' + k] = v.item()
            v.reset()
        # lr = self.scheduler.get_last_lr()[0]
        # log_aml_val('train', 'lr', self.itr, lr, double=False)
        # wandb_dict['train/' + 'lr'] = lr
        # self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item())
        self.timer.reset()
        self.logger.write(log_str)
        if self.wandb_run:
            self.wandb_run.log(wandb_dict, step=self.itr)
        # self.tb_writer.flush()


class Evaluator:

    def __init__(self, opt, train_time=False):

        self.opt = opt

        # set random seed
        # rng = fix_random_seed(opt.get('seed', 2022))
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
            # self.model = PtTransformer(self.opt).cuda()
            self.model = create_model(opt).cuda()
            self.load_model()
            self.model.eval().requires_grad_(False)
        else:
            self.model = None
        pt_gen = opt.pt_gen.clone()
        pt_gen.max_seq_len = opt.model.vid_net.max_seq_len * 10
        self.pt_gen = PtGenerator(**pt_gen).cuda()
        # self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        root = self.opt._root
        # if root.startswith('/tmp/zijia-2024/'):
        #     root = root[len('/tmp/zijia-2024/'):]
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

    def reset(self):
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        self.text_cnt = 0
        self.itr = 0

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
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
        ckpt = utils.Checkpoint(epoch)
        start_time = time.time()
        loss_list = []
        if train_time_data:
            loader = self.dataloader
        else:
            loader = tqdm(self.dataloader)
        for i, data_list in enumerate(loader):

            outputs, results, loss = self.simple_predict(data_list[0])
            targets = data_list[0]['segment']

            # vid, tid = data_list[0]['clip_id'], data_list[0]['text_id']
            # video = utils.Video(vid)
            # video.tid = tid
            # # video.targets = utils.to_numpy(targets)
            # video.abs_target = data_list[0]['segment']
            # video.rel_target = data_list[0]['target']
            # video.results = results
            # video.loss = loss
            # video.outputs = self.outputs # [ [utils.to_numpy(x) for x in r] for r in self.outputs ]
            # ckpt.add_videos(video)

            # print(results)
            # refs = self.predict(data_list[0])
            # assert all([ torch.allclose(results[x]['segments'], refs[x]['segments']) for x in range(len(results)) ])
            # assert all([ torch.allclose(results[x]['scores'], refs[x]['scores']) for x in range(len(results)) ])
            # import ipdb; ipdb.set_trace()


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
                # ref_iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                # assert np.allclose(iou_n, ref_iou_n)

                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            self.text_cnt += len(targets)

            loss_list.append(loss)

            self.itr += 1

            if self.opt.aux.dryrun:
                break

        # ckpt.save('tmp.pk')
        # import ipdb; ipdb.set_trace()
        
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
                # log_aml_val('eval', f"R@{rank}-{thresh:.1f}", writer_it, metrics[i, j], double=False)
                wandb_dict[f"eval/R@{rank}-{thresh:.1f}"] = metrics[i, j]
        
        log_str += "\n-----\n"
        loss_dict = easy_reduce(loss_list, 'mean')
        for k, v in loss_dict.items():
            log_str += f"{k}: {v:.3f}; "
            # log_aml_val('eval', k, writer_it, v, double=False)
            wandb_dict[f"eval/{k}"] = v

        self.logger.write(log_str)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")
        if train_time_data:
            wandb_dict[f"eval/epoch"] = int(epoch)
            # wandb_run.log(wandb_dict, step=writer_it)

        # if save_ckpt:
        #     root = self.opt._root
        #     os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        #     ckpt.save(os.path.join(root, 'models', f'ckpt-{epoch}-{writer_it}.pk'))

    def simple_predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        outputs = self._forward(data)
        loss = self._calc_loss(data, outputs)
        results = self._generate_proposals(data, outputs)
        return outputs, results, loss

    def _forward(self, data):

        assert self.window_size is None, "sliding-window evaluation is not supported"
        assert self.window_stride is None, "sliding-window evaluation is not supported"

        # self.model._vname = data['clip_id']

        s1 = time.perf_counter()

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
        shallow_vid = data['shallow_vid']
        vid_len = vid.size(-1)


        # external scores (n, t)
        ext_scores = data['ext_scores'] # None
        if ext_scores is not None and ext_scores.ndim == 1:
            assert False
            ext_scores = ext_scores[None]

        input_vid_len = self.input_vid_len
        if vid_len > input_vid_len:
            # pad video features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size * self.vid_stride
            # stride = self.opt.model.vid_net.mha_win_size 
            input_vid_len = (vid_len + (stride - 1)) // stride * stride

        t2 = time.perf_counter()
        self.time_dict['prepare'].append(t2 - s1)

        # window, window_offset, window_ext = windows[0], window_offsets[0], window_ext_scores[0]
        window = vid
        shallow_window = shallow_vid
        window_offset = 0
        window_ext = ext_scores

        t1 = time.perf_counter()
        window = F.pad(window, (0, input_vid_len - vid_len))[None]
        shallow_window = F.pad(shallow_window, (0, input_vid_len - vid_len))[None]
        window_mask = torch.arange(input_vid_len).view(1, -1) < vid_len
        window = window.cuda(non_blocking=True)
        shallow_window = shallow_window.cuda(non_blocking=True)
        window_mask = window_mask.cuda(non_blocking=True)
        if window_ext is not None:
            window_ext = F.pad(window_ext, (0, input_vid_len - vid_len))
            window_ext = window_ext.cuda(non_blocking=True)

        # if 'mv_data' in data:
        #     mv_data = data['mv_data']
        #     mv_data = [ [d.cuda(non_blocking=True)] for d in mv_data ]
        #     window = self.model.encode_mv(mv_data, window)
        # mv_data = batchify_mv_and_cuda([data], mode='feature', input_vid_len=input_vid_len)
        # window = self.model.prepare_vid(window, mv_data)
        
        text_cls = data['text_cls'].cuda(non_blocking=True)

        ref = self.model(window, shallow_window, window_mask, text_list, text_cls, text_mask_list, eval=True)
        # for i in range(len(fpn_logits_list)):
        #     for j in range(8):
        #         assert torch.allclose(fpn_logits_list[i][j], ref[0][i][j]), (i, j)
        #         assert torch.allclose(fpn_offsets_list[i][j], ref[1][i][j]), (i, j)
        fpn_logits_list, fpn_offsets_list, fpn_masks_list = ref
        fpn_n_points = [m.size(-1) for m in fpn_masks_list[0]]
        fpn_points = self.pt_gen(fpn_n_points)
        # tmp1 = torch.stack(text_list)
        # tmp2 = torch.stack(text_mask_list)
        # ref, ref2 = self.model.fuse_and_predict(fpn, fpn_masks, tmp1, tmp2)

        fpn_masks_list = [ [m.squeeze(1) for m in fpn_masks] for fpn_masks in fpn_masks_list ]

        t2 = time.perf_counter()
        self.time_dict['forward'].append(t2 - t1)

        self.outputs = [ fpn_logits_list, fpn_offsets_list, fpn_points, fpn_masks_list ] 

        return self.outputs


    def _calc_loss(self, data, outputs):
        fpn_logits_list, fpn_offsets_list, fpn_points, fpn_masks_list = outputs

        points = torch.cat(fpn_points, dim=0)
        # fpn_logits = torch.cat(fpn_logits_list, dim=1)   # (bs, p)
        # fpn_offsets = torch.cat(fpn_offsets_list, dim=1) # (bs, p, 2)
        # fpn_masks = torch.cat(fpn_masks, dim=1)  
        
        targets = data['target'] / self.vid_stride
        targets = targets.cuda(non_blocking=True)
        center_sampling = self.opt['train'].get('center_sampling', 'radius')
        center_sampling_radius = self.opt['train']['center_sampling_radius']
        stats = []
        for i, target in enumerate(targets):
            l, o, _ = annotate_points_per_video(points, target, center_sampling=center_sampling, center_sampling_radius=center_sampling_radius)
            l = l[None]
            o = o[None]

            logits = torch.cat(fpn_logits_list[i], dim=1)
            offsets = torch.cat(fpn_offsets_list[i], dim=1)

            masks = torch.cat(fpn_masks_list[i], dim=1)
            pos_masks = torch.logical_and(l, masks)

            norm = max( pos_masks.sum().item(), 1 )

            cls_loss = calc_focal_loss(logits[masks], l[masks], reduction='sum') / norm
            reg_loss = calc_iou_loss(offsets[pos_masks], o[pos_masks], reg_loss='iou', reduction='sum') / norm

            stats.append({'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item()})
        
        stats = easy_reduce(stats, 'mean', skip_nan=True)
        return stats

    def _generate_proposals(self, data, outputs, window_ext=None, window_offset=0, idx=None):

        fpn_logits_list, fpn_offsets_list, fpn_points, fpn_masks_list = outputs 
        if idx is not None:
            fpn_logits_list = [ [x[idx]] for x in fpn_logits_list]
            fpn_offsets_list = [ [x[idx]] for x in fpn_offsets_list]
            fpn_points = [ fpn_points[idx] ]
            fpn_masks_list = [ [x[idx]] for x in fpn_masks_list]


        t2 = time.perf_counter()

        # collect segments and their scores
        window_segs_list, window_scores_list = tuple(), tuple()
        for idx, (fpn_logits, fpn_offsets) in enumerate(zip(fpn_logits_list, fpn_offsets_list)): # for proposals of each text
            window_segs, window_scores = self._collect_segments(
                fpn_points, fpn_logits, fpn_offsets, fpn_masks_list[idx], 
                window_ext[idx] if window_ext is not None else None
            )
            window_segs += window_offset / self.vid_stride
            window_segs_list += (window_segs.cpu(), )
            window_scores_list += (window_scores.cpu(), )

        # segs_list += (window_segs_list, )
        # scores_list += (window_scores_list, )
        # segs_list.append(window_segs_list)
        # scores_list.append(window_scores_list)
        t3 = time.perf_counter()
        self.time_dict['post_process'].append(t3 - t2)

        # segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        # scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]
        # equal to segs_list = segs_list[0], scores_list = scores_list[0]
        segs_list = window_segs_list
        scores_list = window_scores_list

        t1 = time.perf_counter()
        results = []
        for segs, scores in zip(segs_list, scores_list):
            # # only keep top-k scoring boxes
            # n_topk = min(len(segs), self.pre_nms_topk)
            # idx = scores.argsort(descending=True)[:n_topk]
            # segs = segs[idx]
            # scores = scores[idx]

            # NMS
            # import ipdb; ipdb.set_trace()
            all_segs, all_scores = segs, scores
            segs, scores = self.batched_nms(segs, scores)

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results.append({'segments': segs, 'scores': scores}, )

        t2 = time.perf_counter()
        self.time_dict['nms'].append(t2 - t1)

        return results

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            # scores = scores - scores.min() / (scores.max() - scores.min() + 1e-6)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        # left = pt_ctr - 1.0 * points[:, 3]
        # right = pt_ctr + 1.0 * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

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
            # ref_iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
            # assert np.allclose(iou_n, ref_iou_n)

            # ct += (iou_n[:, None] >= self.iou_threshs[None])
            self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            iou_list.append(iou_n)
        self.text_cnt += len(targets)
        return iou_list

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)