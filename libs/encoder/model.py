from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.checkpoint import checkpoint_sequential

from .video_transformer import SpaceTimeTransformer


def load_egovlp_pretrain_weight(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        weight = ckpt["state_dict"]
    else:
        weight = ckpt

    ckpt = {}
    for w in weight:
        if w == "module.vid_proj.0.weight":
            w2 = "head.weight"
        elif w == "module.vid_proj.0.bias":
            w2 = "head.bias"
        elif w.startswith("module.video_model"):
            w2 = w.replace("module.video_model.", "")
        elif w.startswith("module.text_model"):
            continue
        else:
            print(w)
            continue
        ckpt[w2] = weight[w]
    model_weights = model.state_dict()

    new_ckpt = {}
    for k, v in ckpt.items():
        temporal_embed_name = "temporal_embed"

        if (
            k.startswith("patch_embed.proj")
            and ("img_" + k in model_weights)
            and (model_weights["img_" + k].shape == ckpt[k].shape)
        ):
            print("also load", "img_" + k)
            new_ckpt["img_" + k] = ckpt[k]
        elif (
            k == temporal_embed_name
            and (model_weights[temporal_embed_name].shape[1] < ckpt[k].shape[1])
        ):
            print("also load", temporal_embed_name)
            new_ckpt[temporal_embed_name] = ckpt[k][
                :, : model_weights[temporal_embed_name].shape[1]
            ]
        elif k not in model_weights:
            continue
        elif v.shape == model_weights[k].shape:
            new_ckpt[k] = v
        else:
            print("shape mismatch", k, v.shape, model_weights[k].shape)

    assert len(new_ckpt) > 0
    print("number of weights loaded:", len(new_ckpt))
    print("number of weights in model:", len(model_weights))

    for k in model_weights:
        if k not in new_ckpt:
            print("not loaded:", k)

    model.load_state_dict(new_ckpt, strict=False)
    return list(new_ckpt.keys())


class PoolBlock(nn.Module):
    def __init__(self, opt, pool, block):
        super().__init__()
        self.block = block
        self.pool_type = pool
        self.opt = opt

        if pool in ["temporal", "both"]:
            if opt.encoder.ptype == "conv":
                self.tpool = nn.Conv1d(self.opt.encoder.dim, self.opt.encoder.dim, 3, 2, 1)
            elif opt.encoder.ptype == "avg":
                self.tpool = nn.AvgPool1d(3, 2, 1)

        if pool in ["spatial", "both"]:
            if opt.encoder.ptype == "conv":
                self.spool = nn.Conv2d(self.opt.encoder.dim, self.opt.encoder.dim, 3, 2, 1)
            elif opt.encoder.ptype == "avg":
                self.spool = nn.AvgPool2d(3, 2, 1)

    def forward(self, *args, **kwargs):
        x = args[0]
        npatch = kwargs["time_n"]
        hw = int(np.sqrt(npatch))
        nframe = kwargs["space_f"]

        cls_token = x[:, 0:1]
        x = x[:, 1:]
        if self.pool_type in ["spatial", "both"]:
            x = einops.rearrange(x, "b (t h w) c -> (b t) c h w", t=nframe, h=hw, w=hw)
            x = self.spool(x)
            x = einops.rearrange(x, "(b t) c h w -> b (t h w) c", t=nframe)
            hw = hw // 2

        if self.pool_type in ["temporal", "both"]:
            x = einops.rearrange(x, "b (t h w) c -> (b h w) c t", t=nframe, h=hw, w=hw)
            x = self.tpool(x)
            nframe = x.shape[-1]
            x = einops.rearrange(x, "(b h w) c t -> b (t h w) c", h=hw, w=hw)

        x = torch.cat([cls_token, x], dim=1)
        kwargs["time_n"] = hw**2
        kwargs["space_f"] = nframe
        args = list(args)
        args[0] = x
        return self.block(*args, **kwargs)


class SidekickEncoder(SpaceTimeTransformer):
    def __init__(self, opt):
        self.opt = opt
        nframes = int(opt.data.clip_size / opt.data.clip_d)
        super().__init__(
            img_size=opt.data.img_size,
            patch_size=opt.encoder.patch_size,
            in_chans=3,
            num_classes=opt.encoder.vid_dim,
            embed_dim=opt.encoder.dim,
            depth=opt.encoder.depth,
            num_heads=opt.encoder.n_heads,
            num_frames=nframes,
            grad_ckpt_nseg=opt.encoder.gc_seg,
            opt=opt,
            drop_rate=opt.encoder.drate,
            drop_path_rate=opt.encoder.dprate,
            attn_drop_rate=0.0,
            time_init="zeros",
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            attention_style="frozen-in-time",
        )

        self._feature_extract = None
        self.use_patch_pool = self._use_legacy_patch_pool()
        if self.use_patch_pool:
            # Goalstep legacy checkpoints pooled patches before the transformer.
            self.patch_pool = nn.Conv2d(opt.encoder.dim, opt.encoder.dim, 3, 2, 1)
        self._loaded_keys = load_egovlp_pretrain_weight(self, opt.encoder.pretrain)
        if not self.use_patch_pool:
            self.add_pool()

        self.vid_interp = nn.Conv1d(opt.encoder.dim, opt.encoder.dim * (opt.encoder.fd - 1), 3, 1, 1)
        self.align_map = nn.Linear(opt.encoder.dim, opt.encoder.text_dim)
        self.tau = opt.encoder.tau

    @property
    def vid_model(self):
        # Backward-compat accessor for old worker/eval code paths.
        return self

    def _use_legacy_patch_pool(self):
        sp_loc = self.opt.encoder.sp_loc
        tp_loc = self.opt.encoder.tp_loc
        return sp_loc in [0, "0"] and tp_loc is None

    def add_pool(self):
        if self.opt.encoder.sp_loc is None and self.opt.encoder.tp_loc is None:
            return

        if self.opt.encoder.sp_loc == self.opt.encoder.tp_loc:
            idx = self.opt.encoder.sp_loc
            self.blocks[idx] = PoolBlock(self.opt, "both", self.blocks[idx])
            return

        if self.opt.encoder.sp_loc is not None:
            idx = self.opt.encoder.sp_loc
            self.blocks[idx] = PoolBlock(self.opt, "spatial", self.blocks[idx])
        if self.opt.encoder.tp_loc is not None:
            idx = self.opt.encoder.tp_loc
            self.blocks[idx] = PoolBlock(self.opt, "temporal", self.blocks[idx])

    def _add_token_and_position(self, x, curr_frames):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)
        return x, self.patches_per_frame, curr_frames

    def _prepare_inputs_with_patch_pool(self, x, curr_frames):
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        x = x + tile_pos_embed

        hw = int(np.sqrt(self.patches_per_frame))
        x = einops.rearrange(x, "b (t h w) c -> (b t) c h w", h=hw, w=hw, t=curr_frames)
        x = self.patch_pool(x)
        x = einops.rearrange(x, "(b t) c h w -> b (t h w) c", t=curr_frames)

        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame // 4, 1)
        x = x + tile_temporal_embed

        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        cls_tokens = cls_tokens + cls_embed
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        return x, self.patches_per_frame // 4, curr_frames

    def _feature_extract_forward(self, frames, f2t):
        batch_size = self._feature_extract
        num_batch = (f2t.shape[0] + batch_size - 1) // batch_size

        hidden = []
        for bidx in range(num_batch):
            f2t_tmp = f2t[bidx * batch_size : bidx * batch_size + batch_size]
            b, curr_frames = f2t_tmp.shape[:2]

            x = frames[f2t_tmp]
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(2, 1)
            x = x.reshape(b, -1, self.patch_embed.embed_dim)
            if self.use_patch_pool:
                x, n, f = self._prepare_inputs_with_patch_pool(x, curr_frames)
            else:
                x, n, f = self._add_token_and_position(x, curr_frames)

            h = self.forward_features(x, n, f)
            hidden.append(h)

        hidden = torch.cat(hidden, dim=0)
        x = self.head(hidden)
        return hidden, x

    def prepare_inputs(self, frames, f2t):
        b, curr_frames = f2t.shape[:2]
        frame_patch = self.patch_embed(frames.unsqueeze(0))
        x = frame_patch[f2t]
        x = einops.rearrange(x, "b t c h w -> b (t h w) c")
        if self.use_patch_pool:
            return self._prepare_inputs_with_patch_pool(x, curr_frames)
        return self._add_token_and_position(x, curr_frames)

    def forward_features(self, x, n, f):
        if self.grad_ckpt_nseg is None:
            for blk in self.blocks:
                x = blk(
                    x,
                    self.einops_from_space,
                    self.einops_to_space,
                    self.einops_from_time,
                    self.einops_to_time,
                    time_n=n,
                    space_f=f,
                )
        else:
            assert self.grad_ckpt_nseg >= 1
            blocks = []
            for blk in self.blocks:
                blk_func = partial(
                    blk,
                    einops_from_space=self.einops_from_space,
                    einops_to_space=self.einops_to_space,
                    einops_from_time=self.einops_from_time,
                    einops_to_time=self.einops_to_time,
                    time_n=n,
                    space_f=f,
                )
                if isinstance(blk, PoolBlock):
                    if blk.pool_type in ["spatial", "both"]:
                        n = n // 4
                    if blk.pool_type in ["temporal", "both"]:
                        f = f // 2
                blocks.append(blk_func)

            x = checkpoint_sequential(blocks, self.grad_ckpt_nseg, x, use_reentrant=False)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def _forward_video(self, frames, f2t):
        if self._feature_extract is None:
            x, n, f = self.prepare_inputs(frames, f2t)
            hidden = self.forward_features(x, n, f)
            x = self.head(hidden)
            return hidden, x
        return self._feature_extract_forward(frames, f2t)

    def _forward_model(self, frame, f2t, vid_feat, text_cls_tokens):
        hidden, distill_vid_feat = self._forward_video(frame, f2t)

        if self.opt.encoder.itp:
            indices = np.arange(vid_feat.shape[1])
            input_idx = indices[:: self.opt.encoder.fd]
            pred_idx = [x for x in indices if x not in input_idx]
            input_idx = torch.LongTensor(input_idx).to(frame.device)
            pred_idx = torch.LongTensor(pred_idx).to(frame.device)

            hidden_interp = self.vid_interp(hidden.T)
            hidden_interp = torch.chunk(hidden_interp, self.opt.encoder.fd - 1, 0)
            hidden_interp = torch.stack(hidden_interp, dim=-1)
            hidden_interp = hidden_interp.view(hidden.size(1), -1).T
            hidden_interp = self.norm(hidden_interp)
            hidden_interp = hidden_interp[: len(pred_idx)]
            full_hidden = torch.zeros([vid_feat.shape[1], hidden.shape[1]], device=hidden.device)
            full_hidden[input_idx] = hidden
            full_hidden[pred_idx] = hidden_interp

            hidden = full_hidden
            distill_vid_feat = self.head(hidden)

        vid_feat = self.align_map(hidden)
        vid_feat = vid_feat / vid_feat.norm(dim=1, keepdim=True)
        text_feat = text_cls_tokens / text_cls_tokens.norm(dim=1, keepdim=True)

        vid_text_sim = vid_feat @ text_feat.T / self.tau
        return hidden, vid_feat, vid_text_sim, distill_vid_feat

    def _eval(self, vid_text_simi, label):
        vmask = label.sum(1) > 0
        s = vid_text_simi[vmask].detach().cpu().float().numpy()
        l = label[vmask].detach().cpu().float().numpy()
        try:
            vmap = average_precision_score(l.T, s.T, average="macro")
        except Exception:
            vmap = 0

        tmask = label.sum(0) > 0
        s = vid_text_simi[:, tmask].T.detach().cpu().float().numpy()
        l = label[:, tmask].T.detach().cpu().float().numpy()
        try:
            tmap = average_precision_score(l.T, s.T, average="macro")
        except Exception:
            tmap = 0

        return {"vid_mAP": vmap, "txt_mAP": tmap}

    def _loss(self, vid_text_sim, label, distill_vid_feat, vid_feat):
        def cross_entropy(exp, label):
            if self.opt.encoder.clv == 1:
                denom = exp.sum(1)
                nom = (exp * label).sum(-1)
                loss = -torch.log(nom / (denom + 1e-4))
                loss = loss.mean()
            elif self.opt.encoder.clv == 2:
                neg_sum = exp.sum(-1, keepdims=True) - (exp * label).sum(-1, keepdims=True)
                loss = -torch.log(exp / (neg_sum + exp))
                loss = (loss * label).sum() / label.sum()
            else:
                raise ValueError()
            return loss

        exp_logit = vid_text_sim.exp()

        vmask = label.sum(1) > 0
        vids = torch.where(vmask)[0]
        vlabel = label[vids]
        exp = exp_logit[vids]
        loss1 = cross_entropy(exp, vlabel)

        tmask = label.sum(0) > 0
        tids = torch.where(tmask)[0]
        tlabel = label[:, tids].T
        exp = exp_logit[:, tids].T
        loss2 = cross_entropy(exp, tlabel)

        loss = loss1 + loss2
        if torch.isnan(loss):
            loss = torch.nan_to_num(loss, nan=0.0)
            print(loss1, loss2)

        ref_feat = vid_feat.T
        distill_loss = F.mse_loss(distill_vid_feat, ref_feat)

        return {
            "total_loss": self.opt.encoder.cw * loss + self.opt.encoder.dw * distill_loss,
            "vid_contrast": loss1,
            "text_contrast": loss2,
            "distill": distill_loss,
        }

    def forward(
        self,
        frame,
        f2t=None,
        vid_feat=None,
        text_cls_token=None,
        label=None,
        compute_loss=True,
        evaluate=True,
    ):
        # Backward-compatible two-arg video forward.
        if vid_feat is None and text_cls_token is None and label is None:
            return self._forward_video(frame, f2t)

        v_hidden, v_align, vt_simi, v_distill = self._forward_model(frame, f2t, vid_feat, text_cls_token)
        outputs = [v_hidden, v_align, vt_simi, v_distill]

        if compute_loss:
            loss_dict = self._loss(vt_simi, label, v_distill, vid_feat)
        else:
            loss_dict = {}

        if evaluate:
            loss_dict.update(self._eval(vt_simi, label))

        return outputs, loss_dict

    def load_state_dict(self, state_dict, strict=True):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("vid_model."):
                key = key[len("vid_model.") :]
            normalized[key] = value
        return super().load_state_dict(normalized, strict=strict)


# Backward-compatible aliases for older imports.
SaliencyModel = SidekickEncoder
SpaceTimeTransformer_SharePatch = SidekickEncoder
SpaceTimeTransformerSharePatch = SidekickEncoder
