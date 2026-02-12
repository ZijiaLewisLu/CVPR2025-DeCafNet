import torch
import torch.nn as nn
import einops

from ..modeling.fusion import make_fusion
from ..modeling.head import make_head
from ..modeling.text_net import make_text_net
from ..modeling.video_net import make_video_net
from ..modeling.blocks import MaskedConv1D
from ..modeling.blocks import masked_max_pool1d
from ..modeling.cnn import TemporalCNN
from ..utils.model_utils import PtGenerator



class DecafNetGrounder(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        self.channel_drop = nn.Dropout1d(opt.model.vid_net.cdrop)
        
        self.opt = opt
        self.text_net = make_text_net(opt.model['text_net'])

        in_dim = opt.model.vid_net.in_dim
        self.vid_map = MaskedConv1D(in_dim, opt.model.vid_net.embd_dim, 1)
        _opt = opt.model.vid_net.clone()
        _opt.in_dim = _opt.embd_dim
        self.vid_net = make_video_net(_opt)

        # fusion and prediction heads
        self.fusion = make_fusion(opt.model['fusion'])
        self.cls_head  = make_head(opt.model['cls_head'])
        self.refine = TemporalCNN(self.opt.model.vid_net.arch[-1], 
                            self.opt.model.rdim, self.opt.model.rdim,
                            num_layers=self.opt.model.vid_net.arch[-1] if self.opt.model.rlayer is None else self.opt.model.rlayer,
                            in_map=True)
        opt.model.cls_head.embd_dim += self.opt.model.rdim
        self.cls_head2 = make_head(opt.model['cls_head'])
        opt.model.reg_head.embd_dim += self.opt.model.rdim
        self.reg_head = make_head(opt.model['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        
        # expand
        expand_fpn_logits = [fpn_logits[0]]
        ref_len = fpn_logits[0].shape[1]
        for i, l in enumerate(fpn_logits[1:]):
            new_l = torch.nn.functional.interpolate(l.unsqueeze(1), size=ref_len, mode='nearest')[:, 0]
            new_l = new_l * fpn_masks[0][:, 0]
            expand_fpn_logits.append(new_l)
        expand_fpn_logits = torch.stack(expand_fpn_logits, dim=1) 

        # refine
        expand_fpn_logits = self.refine(expand_fpn_logits, fpn_masks[0])

        # downsample
        new_fpn = []
        for i, f in enumerate(fpn):
            if i != 0:
                expand_fpn_logits = masked_max_pool1d(expand_fpn_logits, fpn_masks[i-1])[0]
            f = torch.cat([f, expand_fpn_logits], dim=1)
            new_fpn.append(f)

        fpn_logits2, _ = self.cls_head2(new_fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(new_fpn, fpn_masks)
        return fpn_logits, fpn_logits2, fpn_offsets, fpn_masks

    def forward(self, vid, vid_masks, text, text_cls, text_masks, text_size=None, evaluate=False):
        if evaluate:
            return self._eval_forward(vid, vid_masks, text, text_masks, text_size)
        return self._train_forward(vid, vid_masks, text, text_masks, text_size)

    def _eval_forward(self, vid, vid_masks, text, text_masks, text_size=None):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        vid = self.channel_drop(vid)
        vid_masks = vid_masks.unsqueeze(1)
        vid, vid_masks = self.vid_map(vid, vid_masks)

        text_list = text
        text_mask_list = text_masks
        fpn_logits_list = []
        fpn_offsets_list = []
        fpn_masks_list = []       #     vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe

        for b, (text, text_masks) in enumerate(zip(text_list, text_mask_list)):
            vid_tmp, vid_masks_tmp = self.fusion(vid, vid_masks, text, text_masks, text_size)
            fpn_tmp, fpn_masks_tmp = self.encode_video(vid_tmp, vid_masks_tmp)
            l1, l2, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_masks) # tuple, length=the number of FPN
            fpn_logits_list.append(l2)
            fpn_offsets_list.append(o)
            fpn_masks_list.append(m)

        return fpn_logits_list, fpn_offsets_list, fpn_masks_list

    def _train_forward(self, vid, vid_masks, text, text_masks, text_size=None):
        """
        vid: b, h, t
        vid_masks: b, t
        """

        vid = self.channel_drop(vid)
        vid_masks = vid_masks.unsqueeze(1)
        vid, vid_masks = self.vid_map(vid, vid_masks)

        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        vid, vid_masks = self.fusion(vid, vid_masks, text, text_masks, text_size)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks

class DecafNetGrounderLateFusion(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        self.channel_drop = nn.Dropout1d(opt.model.vid_net.cdrop)
        
        self.opt = opt
        self.text_net = make_text_net(opt.model['text_net'])

        self.vid_net = make_video_net(opt.model['vid_net'])

        # fusion and prediction heads
        self.fusion = make_fusion(opt.model['fusion'])
        self.cls_head  = make_head(opt.model['cls_head'])
     
        self.refine = TemporalCNN(self.opt.model.vid_net.arch[-1], 
                            self.opt.model.rdim, self.opt.model.rdim,
                            num_layers=self.opt.model.vid_net.arch[-1] if self.opt.model.rlayer is None else self.opt.model.rlayer,
                            in_map=True)
        opt.model.cls_head.embd_dim += self.opt.model.rdim
        self.cls_head2 = make_head(opt.model['cls_head'])
        opt.model.reg_head.embd_dim += self.opt.model.rdim
        self.reg_head = make_head(opt.model['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        
        # expand
        expand_fpn_logits = [fpn_logits[0]]
        ref_len = fpn_logits[0].shape[1]
        for i, l in enumerate(fpn_logits[1:]):
            new_l = torch.nn.functional.interpolate(l.unsqueeze(1), size=ref_len, mode='nearest')[:, 0]
            new_l = new_l * fpn_masks[0][:, 0]
            expand_fpn_logits.append(new_l)
        expand_fpn_logits = torch.stack(expand_fpn_logits, dim=1) 

        # refine
        expand_fpn_logits = self.refine(expand_fpn_logits, fpn_masks[0])

        # downsample
        new_fpn = []
        for i, f in enumerate(fpn):
            if i != 0:
                expand_fpn_logits = masked_max_pool1d(expand_fpn_logits, fpn_masks[i-1])[0]
            f = torch.cat([f, expand_fpn_logits], dim=1)
            new_fpn.append(f)

        fpn_logits2, _ = self.cls_head2(new_fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(new_fpn, fpn_masks)
        return fpn_logits, fpn_logits2, fpn_offsets, fpn_masks

    def forward(self, vid, vid_masks, text, text_cls, text_masks, text_size=None, evaluate=False):
        if evaluate:
            return self._eval_forward(vid, vid_masks, text, text_masks, text_size)
        return self._train_forward(vid, vid_masks, text, text_masks, text_size)

    def _eval_forward(self, vid, vid_masks, text, text_masks, text_size=None):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        vid = self.channel_drop(vid)
        vid_masks = vid_masks.unsqueeze(1)

        fpn, fpn_masks = self.encode_video(vid, vid_masks)

        text_list = text
        text_mask_list = text_masks
        fpn_logits_list = []
        fpn_offsets_list = []
        fpn_masks_list = []       #     vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe

        for b, (text, text_masks) in enumerate(zip(text_list, text_mask_list)):
            fpn_tmp, fpn_masks_tmp = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
            l1, l2, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_masks) # tuple, length=the number of FPN
            fpn_logits_list.append(l2)
            fpn_offsets_list.append(o)
            fpn_masks_list.append(m)

        return fpn_logits_list, fpn_offsets_list, fpn_masks_list

    def _train_forward(self, vid, vid_masks, text, text_masks, text_size=None):
        """
        vid: b, h, t
        vid_masks: b, t
        """

        vid = self.channel_drop(vid)
        vid_masks = vid_masks.unsqueeze(1)

        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)

        fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks
