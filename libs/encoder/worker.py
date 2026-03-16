"""
Training encoder only
encoder has temporal interpolation and adaptor
"""
from contextlib import nullcontext
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .dataset import make_dataset, make_dataloader
from .model import SidekickEncoder
from ..utils.dist_utils import get_rank, get_world_size, barrier, print0, get_local_rank
from ..modeling.optim import make_encoder_optimizer
from ..utils.train_utils import Logger, fix_random_seed, log
from ..utils.opt_utils import to_numpy, count_parameters

import torch.distributed as dist
import pickle as pk
from torch.cuda.amp import autocast, GradScaler


def try_catch(func, t=15, msg=""):
    for i in range(t):
        try:
            return func()
        except Exception as e:
            print(f"Rank{get_rank()} - Attempt {i} {msg} | {e}")
            if i == t - 1:
                raise e


def handle_scalar_array(array):
    if isinstance(array, torch.Tensor):
        return array.detach().item()
    if isinstance(array, np.ndarray):
        return array.item()
    return array


def batchify_and_cuda(data_list):
    assert len(data_list) == 1
    frame = data_list[0]["frame"].cuda(non_blocking=True)
    vid_feat = data_list[0]["vid"].cuda(non_blocking=True)
    f2t = data_list[0]["f2t"].cuda(non_blocking=True)

    text_dict = {}
    for d in data_list:
        text_dict.update(d["text_cls_dict"])
    text_keys = list(text_dict.keys())
    text_cls_tokens = torch.stack([text_dict[k] for k in text_keys], dim=0).cuda(
        non_blocking=True
    )
    text2id = {k: i for i, k in enumerate(text_keys)}

    label = torch.zeros(vid_feat.shape[1], len(text2id)).long()
    text_label = data_list[0]["text_label"]
    foffset = 0
    for i, texts in text_label.items():
        if texts is None:
            continue
        for text in texts:
            label[i + foffset, text2id[text]] = 1
    label = label.cuda(non_blocking=True)

    return frame, f2t, vid_feat, text_cls_tokens, label


class Trainer:

    def __init__(self, opt, wandb_run=None):
        self.opt = opt

        assert self.opt.model.on is False

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        print("Building model...")
        self.model = SidekickEncoder(opt).cuda()
        print('Model Size', count_parameters(self.model, scale=1e6))

        # prepare dataset
        print0("Building dataset...")
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(opt, num_epochs=self.num_epochs, is_training=True)

        print0("Building loader...")
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt.train.num_workers,
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('gacm', opt['train'].get('microbatch_size', 1))
        print0('Epoch Size:', len(self.dataloader))

        # build training utilities
        self.itrs_per_epoch = opt['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        if self.opt.model.pretrain is None or self.opt.optimizer.lpretrain is None:
            lr_group = [ [opt.optimizer.lr, list(self.model.named_parameters())] ]
        else:
            pretrained_weight = []
            new_weight = []
            for n, p in self.model.named_parameters():
                if n in self.model._loaded_keys:
                    pretrained_weight.append((n, p))
                else:
                    new_weight.append((n, p))
            lr_group = [ [opt.optimizer.lr, new_weight], [opt.optimizer.lpretrain, pretrained_weight] ]

        self.optimizer = make_encoder_optimizer(self.model, opt['optimizer'], lr_group)
        self.clip_grad_norm = opt['optimizer'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt.aux.log_interval
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.logger.write("-------------------------------------------------------------------\n")
            self.logger.write(f'seed {opt.seed}')
        else:
            self.logger = None

        # load model weights and training states
        if opt.aux.model_resume is not None:
            print('Init model weight from', opt.aux.model_resume)
            model_ckpt = torch.load(opt.aux.model_resume, map_location='cpu')
            self.model.load_state_dict(model_ckpt['model'])
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_local_rank()], find_unused_parameters=False)

        self.evaluator = Evaluator(self.opt, wandb_run=wandb_run)
        self.wandb_run = wandb_run

        self.scaler = GradScaler()

    def run(self):
        print("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)

            start_time = time.time()

            log_results = []
            for data_list in self.dataloader:
                with autocast(dtype=torch.bfloat16):
                    frame, f2t, vid_feat, text_cls_token, label = batchify_and_cuda(data_list)
                    if self.itr % self.microbatch_size == 0:
                        _, loss_dict = self.model(frame, f2t, vid_feat, text_cls_token, label)
                        total_loss = loss_dict['total_loss'] / self.microbatch_size
                        self.scaler.scale(total_loss).backward()
                        if self.clip_grad_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        sync_context = self.model.no_sync if self.opt['_distributed'] else nullcontext
                        with sync_context():
                            _, loss_dict = self.model(frame, f2t, vid_feat, text_cls_token, label)
                            total_loss = loss_dict['total_loss'] / self.microbatch_size
                            self.scaler.scale(total_loss).backward()
                            if self.clip_grad_norm:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.itr += 1

                if self.opt.aux.dryrun:
                    break

                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    loss_dict = { k: handle_scalar_array(v) for k, v in loss_dict.items() }
                    loss_dict['time'] = time.time() - start_time
                    loss_dict['progress'] = self.itr / self.num_itrs
                    log_results.append(loss_dict)
                    start_time = time.time()

                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        log_str = log(self.itr, log_results, self.wandb_run, prefix='train')
                        self.logger.write(f"[{self.itr}/{self.num_itrs}] {log_str}")
                        log_results = []

                if (self.opt.aux.eval_run > 0) and (self.itr % self.opt.aux.eval_run == 0):
                    if get_rank() == 0:
                        print('Saving checkpoint')
                        try_catch(self.checkpoint, msg='saving checkpoint')

                    print('EvalRun', get_rank())
                    self.model.eval()
                    self._unwrap(self.model).vid_model._feature_extract = 256
                    self.evaluator.run_during_training(self.model, self.epoch, self.itr)
                    self._unwrap(self.model).vid_model._feature_extract = None
                    self.model.train()
                    barrier()

            if self.opt.aux.dryrun:
                break

            self.epoch += 1
            barrier()

        try_catch(self.checkpoint, msg='model checkpoint')
        print0("Training completed.")

    def load(self, ckpt='last.pth', root=None):
        if root is None:
            root = self.opt['_root']
        model_path = os.path.join(root, 'models', ckpt)
        state_path = os.path.join(root, 'states', ckpt)
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])

        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        if get_rank() != 0:
            return
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(model_ckpt, os.path.join(model_dir, f"{self.epoch}-{self.itr}.pth"))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        print(os.path.join(model_dir, f"{self.epoch}-{self.itr}.pth"))

class Evaluator:

    def __init__(self, opt, wandb_run=None):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt, is_training=False)
        self.dataset = dataset
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=False,
            batch_size=1, num_workers=1,
            world_size=get_world_size(), rank=get_rank()
        )
        self.num_itrs = len(self.dataloader)

        self.wandb_run = wandb_run

    def make_logger(self, ckpt_str=None):
        if ckpt_str is None:
            ckpt_str = self.opt['_ckpt']
        logger = Logger(os.path.join(self.opt['_root'], f"eval_{ckpt_str}.txt"), dump_to_file=True)
        return logger

    def load_model(self, model, ckpt_str=None, ckpt_path=None):
        if ckpt_path is None:
            if ckpt_str is None:
                ckpt_str = self.opt['_ckpt']

            ckpt_path = os.path.join(
                self.opt['_root'], 'models', f"{ckpt_str}.pth"
            )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        print0(f"Loaded checkpoint [epoch {ckpt_str}]...")

    @torch.no_grad()
    def run(self):
        ckpt_str = self.opt['_ckpt']
        logger = self.make_logger(ckpt_str=ckpt_str)
        model = SidekickEncoder(self.opt).cuda()
        self.load_model(model, ckpt_str=ckpt_str)
        model.eval().requires_grad_(False)

        model_data = {
            "epoch": ckpt_str,
            "iteration": 0,
            "model": model,
            "opt": self.opt,
            "logger": logger,
        }
        model_dict = {0: model_data}
        self._run_feature_extraction(model_dict, self.wandb_run, itr_offset=0, repeat=1, save=False)

    @torch.no_grad()
    def run_during_training(self, model, epoch, iteration):
        model_data = {
            "epoch": epoch,
            "iteration": iteration,
            "model": model,
            "opt": self.opt,
            "logger": self.make_logger(ckpt_str=f"{epoch}-{iteration}"),
        }
        model_dict = { iteration: model_data }
        self._run_feature_extraction(model_dict, self.wandb_run, itr_offset=0, repeat=1, save=False)

    def selection_eval(self, v_feat, text_feat, query_rel_loc, ratio, coverage):
        query_rel_loc = query_rel_loc.long()
        correl = torch.matmul(v_feat, text_feat.T)

        ranked = correl.argsort(dim=0)
        T = correl.shape[0]
        for j in range(len(query_rel_loc)):
            label = torch.zeros_like(correl[:, 0])
            s, e = query_rel_loc[j]
            if e == T:
                s = min(s, e-1)
            else:
                e = max(e, s + 1)
            label[s:e] = 1
            if torch.sum(label) == 0:
                continue
            for i in range(len(ratio)):
                topk = ranked[-int(ratio[i] * T):, j]
                v = label[topk].sum().float() / label.sum().float()
                coverage[i].append(v.item())

    @torch.no_grad()
    def _run_feature_extraction(self, model_dict, wandb_run, itr_offset=0, repeat=1, save=False):
        assert repeat == 1
        assert len(model_dict) == 1

        start = time.time()
        if get_rank() == 0:
            dataloader = tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        all_metrics = []

        ratio = [ 0.7, 0.5, 0.3 ]
        coverage_align = [ [] for _ in ratio ]
        coverage_distill = [ [] for _ in ratio ]
        coverage_both = [ [] for _ in ratio ]
        with autocast(dtype=torch.bfloat16):
            for data_list in dataloader:
                assert len(data_list) == 1
                for iteration, model in model_dict.items():
                    frame, f2t, vid_feat, text_cls_token, label = batchify_and_cuda(data_list)
                    preds, loss_dict = model['model'](frame, f2t, vid_feat, text_cls_token, label, compute_loss=True, evaluate=False)
                    v_hidden, v_align, vid_text_sim, v_distill = preds
                    vname = data_list[0]['clip_id']

                    if self.opt._distributed:
                        metrics = model['model'].module._eval(vid_text_sim, label)
                    else:
                        metrics = model['model']._eval(vid_text_sim, label)
                    loss_dict = {k: handle_scalar_array(v) for k, v in loss_dict.items()}
                    loss_dict.update(metrics)
                    all_metrics.append(loss_dict)

                    if text_cls_token.shape[0] == data_list[0]['target'].shape[0]:
                        text_feat = text_cls_token / text_cls_token.norm(dim=1, keepdim=True)
                        self.selection_eval(v_align, text_feat, data_list[0]['target'], ratio, coverage_align)
                        self.selection_eval(v_distill, text_feat, data_list[0]['target'], ratio, coverage_distill)
                        self.selection_eval((v_align+v_distill)/2, text_feat, data_list[0]['target'], ratio, coverage_both)

                    if save:
                        savefname = os.path.join(model['save_fdr'], vname + '.pk')
                        save = [ v_align.float(), v_distill.float(), vid_text_sim.float(), label ]
                        save = [ to_numpy(x) for x in save ]
                        with open(savefname, 'wb') as fp:
                            pk.dump(save, fp)

                if self.opt.aux.dryrun:
                    break

        if save:
            return

        # update standard metrics
        keys = sorted(all_metrics[0].keys())
        values = []
        for k in keys:
            v = sum([m[k] for m in all_metrics])
            values.append(v)
        values.append(len(all_metrics))
        if self.opt._distributed:
            values = torch.tensor(values).cuda()
            dist.all_reduce(values, dist.ReduceOp.SUM, async_op=False)
            values = values.tolist()
        new_metrics = { k:values[i]/values[-1] for i, k in enumerate(keys) }

        # update selection saliency
        values = []
        for i, r in enumerate(ratio):
            values.append(sum(coverage_align[i]))
            values.append(sum(coverage_distill[i]))
            values.append(sum(coverage_both[i]))
        values.append(len(coverage_align[0]))
        if self.opt._distributed:
            values = torch.tensor(values).cuda()
            dist.all_reduce(values, dist.ReduceOp.SUM, async_op=False)
            values = values.tolist()
        for i, r in enumerate(ratio):
            new_metrics[f'SACC_align@{r}'] = values[i*3] / values[-1]
            new_metrics[f'SACC_distill@{r}'] = values[i*3+1] / values[-1]
            new_metrics[f'SACC_both@{r}'] = values[i*3+2] / values[-1]

        if get_rank() == 0:
            log_str = log(iteration+itr_offset, [new_metrics], wandb_run, prefix='eval')
            model['logger'].write(f"\nFinal: {log_str}")

            print0('Evaluation Time', time.time() - start)
