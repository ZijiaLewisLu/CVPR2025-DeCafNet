import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from ..modeling import sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss
from ..nms import batched_nms
from .train_utils import AverageMeter, iou, time_str, easy_reduce


def calc_focal_loss(logits, labels, smoothing=0.2, alpha=0.5, reduction='sum'):
    labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
    return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction=reduction)


def calc_iou_loss(pred_offsets, gt_offsets, reg_loss='diou', reduction='sum'):
    iou_loss = ctr_diou_loss if reg_loss == 'diou' else ctr_giou_loss
    return iou_loss(pred_offsets, gt_offsets, reduction=reduction)


def annotate_points_per_video(points, target, center_sampling='radius', center_sampling_radius=1.5):
    pt2start = points[:, 0] - target[0]
    pt2end = target[1] - points[:, 0]
    offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

    if center_sampling == 'radius':
        ctr = 0.5 * (target[0] + target[1])
        radius = points[:, 3] * center_sampling_radius
        t_min = (ctr - radius).clamp_(min=target[0])
        t_max = (ctr + radius).clamp_(max=target[1])
        pt2left = points[:, 0] - t_min
        pt2right = t_max - points[:, 0]
        inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
    else:
        inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

    max_reg_dist = torch.maximum(pt2start, pt2end)
    inside_range = torch.logical_and(
        max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
    )
    labels = torch.logical_and(inside_window, inside_range)
    return labels, offsets


def batchify_videos(vid_list, input_vid_len):
    bs = len(vid_list)
    vid_dim = vid_list[0].size(0)
    vid_lens = [v.size(-1) for v in vid_list]
    vid = vid_list[0].new_full((bs, vid_dim, input_vid_len), 0.0)
    for idx in range(bs):
        vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
    vid_lens = torch.as_tensor(vid_lens)[:, None]
    vid_masks = torch.arange(input_vid_len)[None] < vid_lens
    return vid, vid_masks


def batchify_text(text_list, max_text_len):
    bs = len(text_list)
    text_dim = text_list[0].size(0)
    text_lens = [t.size(-1) for t in text_list]
    text = text_list[0].new_full((bs, text_dim, max_text_len), 0.0)
    for idx in range(bs):
        text[idx, :, :text_lens[idx]].copy_(text_list[idx])
    text_lens = torch.as_tensor(text_lens)[:, None]
    text_masks = torch.arange(max_text_len)[None] < text_lens
    return text, text_masks


def batchify_video_text(vid_list, text_list, input_vid_len, max_text_len):
    assert len(vid_list) == len(text_list)
    bs = len(vid_list)

    vid, vid_masks = batchify_videos(vid_list, input_vid_len)

    if isinstance(text_list[0], tuple):
        b_text, b_text_masks = tuple(), tuple()
        n = tuple()
        for t in text_list:
            b_t, b_tm = batchify_text(t, max_text_len)
            b_text += (b_t,)
            b_text_masks += (b_tm,)
            n += (len(t),)
        n_max = max(n)

        text_dim = b_text[0].size(1)
        text = b_text[0].new_full((bs, n_max, text_dim, max_text_len), 0.0)
        for idx in range(bs):
            text[idx, :n[idx]].copy_(b_text[idx])

        text_masks = b_text_masks[0].new_full((bs, n_max, max_text_len), 0, dtype=torch.bool)
        for idx in range(bs):
            text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
    else:
        n = bs * (1,)
        text, text_masks = batchify_text(text_list, max_text_len)

    text_size = torch.as_tensor(n)
    return vid, vid_masks, text, text_masks, text_size


def annotate_points(points, targets, center_sampling='radius', center_sampling_radius=1.5):
    labels_list, offsets_list = [], []
    for target in targets:
        labels, offsets = annotate_points_per_video(
            points,
            target,
            center_sampling=center_sampling,
            center_sampling_radius=center_sampling_radius,
        )
        labels_list.append(labels)
        offsets_list.append(offsets)
    labels = torch.stack(labels_list)
    offsets = torch.stack(offsets_list)
    return labels, offsets



class BaseTrainer:
    log_with_lr = False

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)

            start_time = time.time()

            for data_list in self.dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
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

            if self.opt.aux.dryrun:
                break
        print0("Training completed.")

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

        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = self.loss_norm_momentum * self.loss_norm \
                        + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)

        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _batchify_videos(self, vid_list):
        return batchify_videos(vid_list, self.input_vid_len)

    def _batchify_text(self, text_list):
        return batchify_text(text_list, self.max_text_len)

    def _batchify(self, vid_list, text_list):
        return batchify_video_text(
            vid_list, text_list, self.input_vid_len, self.max_text_len
        )

    def _annotate_points(self, points, targets):
        return annotate_points(
            points,
            targets,
            center_sampling=self.center_sampling,
            center_sampling_radius=self.center_sampling_radius,
        )

    def _annotate_points_per_video(self, points, target):
        return annotate_points_per_video(
            points,
            target,
            center_sampling=self.center_sampling,
            center_sampling_radius=self.center_sampling_radius,
        )

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

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        wandb_dict = {}
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            wandb_dict['train/' + k] = v.item()
            v.reset()
        if self.log_with_lr:
            lr = self.scheduler.get_last_lr()[0]
            wandb_dict['train/lr'] = lr
        log_str += time_str(self.timer.item())
        self.timer.reset()
        self.logger.write(log_str)
        if self.wandb_run:
            self.wandb_run.log(wandb_dict, step=self.itr)


class BaseEvaluator:
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


    def simple_predict(self, data):
        outputs = self._forward(data)
        results = self._generate_proposals(data, outputs)
        return outputs, results

    def _generate_proposals(self, data, outputs, window_ext=None, window_offset=0, idx=None):
        fpn_logits_list, fpn_offsets_list, fpn_points, fpn_masks_list = outputs
        if idx is not None:
            fpn_logits_list = [[x[idx]] for x in fpn_logits_list]
            fpn_offsets_list = [[x[idx]] for x in fpn_offsets_list]
            fpn_points = [fpn_points[idx]]
            fpn_masks_list = [[x[idx]] for x in fpn_masks_list]

        t2 = time.perf_counter()

        window_segs_list, window_scores_list = tuple(), tuple()
        for idx, (fpn_logits, fpn_offsets) in enumerate(zip(fpn_logits_list, fpn_offsets_list)):
            window_segs, window_scores = self._collect_segments(
                fpn_points, fpn_logits, fpn_offsets, fpn_masks_list[idx],
                window_ext[idx] if window_ext is not None else None
            )
            window_segs += window_offset / self.vid_stride
            window_segs_list += (window_segs.cpu(),)
            window_scores_list += (window_scores.cpu(),)

        t3 = time.perf_counter()
        self.time_dict['post_process'].append(t3 - t2)

        segs_list = window_segs_list
        scores_list = window_scores_list

        t1 = time.perf_counter()
        results = []
        for segs, scores in zip(segs_list, scores_list):
            segs, scores = self.batched_nms(segs, scores)

            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results.append({'segments': segs, 'scores': scores},)

        t2 = time.perf_counter()
        self.time_dict['nms'].append(t2 - t1)
        return results

    def _collect_segments(
        self,
        fpn_points,
        fpn_logits,
        fpn_offsets,
        fpn_masks,
        ext_scores,
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            idx = scores > self.pre_nms_thresh
            points_list += (points[idx],)
            scores_list += (scores[idx],)
            offsets_list += (offsets[idx],)

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

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
