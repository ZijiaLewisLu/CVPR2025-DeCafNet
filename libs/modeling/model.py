import torch
import torch.nn as nn
import einops

from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net
from .blocks import MaskedConv1D
from .blocks import masked_max_pool1d
from .tcn import TCN
# from ..mv_model.model import build_motion_encoder
# from ..mv_model.fusion import build_input_fusion_model
# from ..dist_utils import print0

def emulate_mask_pooling(vid_masks, fpn):
    """
    vid_masks: b, 1, t
    """
    fpn_masks = []
    mask_float = vid_masks.unsqueeze(1).float()
    for i in range(len(fpn)):
        mask_float = torch.nn.functional.interpolate(
            mask_float, size=fpn[i].size(-1), mode='nearest'
        )
        mask = mask_float.bool()
        fpn_masks.append(mask)
    return fpn_masks

class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        self.channel_drop = nn.Dropout1d(opt.model.vid_net.cdrop)
        
        self.opt = opt
        # opt = opt.model
        # backbones
        self.text_net = make_text_net(opt.model['text_net'])

        _opt = opt.model.vid_net.clone()
        if opt.model.msf:
            _opt.in_dim *= 2
        if opt.model.scat:
            _opt.in_dim += 1
        self.vid_net = make_video_net(_opt)

        # fusion and prediction heads
        self.fusion = make_fusion(opt.model['fusion'])
        self.cls_head = make_head(opt.model['cls_head'])
        self.reg_head = make_head(opt.model['reg_head'])


    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks

    def forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        return self._drop_forward(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)
        # if self.opt.model.sratio < 1:
        #     return self._drop_forward(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)
        # else:
        #     return self._forward(vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)

    def _forward(self, vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        pass

    def _drop_forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        assert mv_data is None


        N = self.opt.model.sn
        ratio = self.opt.model.sratio

        if (not eval) and text_size is not None: # and text_size.size(0) != vid.size(0):
            vid = vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_size, dim=0) # nbatch x ntext, nframe
        if eval and (text_size is None) and (text_cls.size(0) != vid.size(0)):
            vid = vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe
        

        # import ipdb; ipdb.set_trace()
        if self.opt.model.norm:
            v = shallow_vid / (shallow_vid.norm(dim=1, keepdim=True) + 1e-4)
            t = text_cls / (text_cls.norm(dim=1, keepdim=True) + 1e-4)
            correl = torch.einsum('bht,bh->bt', v, t)  # b x num_text, num_frame
        else:
            correl = torch.einsum('bht,bh->bt', shallow_vid, text_cls)  # b x num_text, num_frame
        all_weight = torch.zeros_like(correl)
        for b in range(correl.shape[0]):
            vid_len = vid_masks[b].sum()
            correl_batch = torch.nn.functional.avg_pool1d(correl[b, None, :vid_len], kernel_size=N, stride=N, ceil_mode=True)[0]
            # correl_batch = torch.nn.functional.adaptive_avg_pool1d(correl[b, None, :vid_len], 16)[0]
            ranked = correl_batch.argsort()
            topk = ranked[-int(ratio * correl_batch.shape[0]):]

            weight = torch.zeros_like(correl_batch)
            weight[topk] = 1
            weight = torch.nn.functional.interpolate(weight[None, None, :], size=vid_len, mode='nearest')[0, 0]
            all_weight[b, :vid_len] = weight
        
        vid = vid * all_weight.unsqueeze(1)
        if not self.opt.model.msf:
            vid_masks = torch.logical_and(all_weight.bool(), vid_masks)
        else:
            vid = torch.cat([vid, shallow_vid], dim=1)
        if self.opt.model.scat:
            vid = torch.cat([vid, correl.unsqueeze(1)], dim=1)

        vid = self.channel_drop(vid)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)

        if not eval:
            if text.ndim == 4:
                text = torch.cat([t[:k] for t, k in zip(text, text_size)])
            if text_masks.ndim == 3:
                text_masks = torch.cat(
                    [t[:k] for t, k in zip(text_masks, text_size)]
                )
            
            text, text_masks = self.encode_text(text, text_masks)
            fpn_logits, fpn_offsets, fpn_masks = \
                self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

            return fpn_logits, fpn_offsets, fpn_masks

        else:
            text_list = text
            text_mask_list = text_masks
            fpn_logits_list = []
            fpn_offsets_list = []
            fpn_masks_list = []
            # if self.opt.model.sratio < 1:
            for i, (text, text_mask) in enumerate(zip(text_list, text_mask_list)):
                fpn_tmp = tuple([ x[i:i+1] for x in fpn ])
                fpn_masks_tmp = tuple([ x[i:i+1] for x in fpn_masks ])
                l, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_mask) # tuple, length=the number of FPN
                fpn_logits_list.append(l)
                fpn_offsets_list.append(o)
                fpn_masks_list.append(m)
            return fpn_logits_list, fpn_offsets_list, fpn_masks_list

class PtTransformerEarlyFusion(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt, second_fusion=True):
        super().__init__()

        self.channel_drop = nn.Dropout1d(opt.model.vid_net.cdrop)
        
        self.opt = opt
        self.text_net = make_text_net(opt.model['text_net'])

        # _opt = opt.model.vid_net.clone()
        in_dim = opt.model.vid_net.in_dim
        if opt.model.msf:
            in_dim *= 2
        if opt.model.scat:
            in_dim += 1
        # self.vid_map = nn.Linear(in_dim, opt.model.vid_net.in_dim)
        self.vid_map = MaskedConv1D(in_dim, opt.model.vid_net.embd_dim, 1)
        _opt = opt.model.vid_net.clone()
        _opt.in_dim = _opt.embd_dim
        self.vid_net = make_video_net(_opt)

        # fusion and prediction heads
        self.fusion = make_fusion(opt.model['fusion'])
        self.cls_head = make_head(opt.model['cls_head'])
        self.reg_head = make_head(opt.model['reg_head'])

        self.second_fusion = second_fusion

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        if self.second_fusion:
            fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks

    def forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        if eval:
            return self._drop_forward_eval(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)

        return self._drop_forward(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)


    def _drop_forward_eval(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        assert mv_data is None
        assert eval


        N = self.opt.model.sn
        ratio = self.opt.model.sratio

        # if (not eval) and text_size is not None: # and text_size.size(0) != vid.size(0):
        #     vid = vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
        #     shallow_vid = shallow_vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
        #     vid_masks = vid_masks.repeat_interleave(text_size, dim=0) # nbatch x ntext, nframe
        assert vid.size(0) == 1, vid.size()
        # if eval and (text_size is None) and (text_cls.size(0) != vid.size(0)):
        #     vid = vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
        #     shallow_vid = shallow_vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
        if self.opt.model.norm:
            v = shallow_vid / (shallow_vid.norm(dim=1, keepdim=True) + 1e-4)
            t = text_cls / (text_cls.norm(dim=1, keepdim=True) + 1e-4)
            correl = torch.einsum('bht,bh->bt', v, t)  # b x num_text, num_frame
        else:
            correl = torch.einsum('bht,bh->bt', shallow_vid, text_cls)  # b x num_text, num_frame

        vid_origin = vid
        vid_masks_origin = vid_masks

        text_list = text
        text_mask_list = text_masks
        fpn_logits_list = []
        fpn_offsets_list = []
        fpn_masks_list = []       #     vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe
        for b, (text, text_masks) in enumerate(zip(text_list, text_mask_list)):
            vid = vid_origin.clone()
            vid_masks = vid_masks_origin.clone()
        
            # all_weight = torch.zeros_like(correl)
            vid_len = vid_masks.sum()
            correl_batch = torch.nn.functional.avg_pool1d(correl[b, None, :vid_len], kernel_size=N, stride=N, ceil_mode=True)[0]
            # correl_batch = torch.nn.functional.adaptive_avg_pool1d(correl[b, None, :vid_len], 16)[0]
            ranked = correl_batch.argsort()
            topk = ranked[-int(ratio * correl_batch.shape[0]):]

            weight = torch.zeros_like(correl_batch)
            weight[topk] = 1
            weight = torch.nn.functional.interpolate(weight[None, None, :], size=vid_len, mode='nearest')[0, 0]
            all_weight = torch.zeros_like(vid_masks)
            all_weight[0, :vid_len] = weight
            
            vid = vid * all_weight.unsqueeze(1)
            if not self.opt.model.msf:
                vid_masks = torch.logical_and(all_weight.bool(), vid_masks)
            else:
                vid = torch.cat([vid, shallow_vid], dim=1)
            if self.opt.model.scat:
                vid = torch.cat([vid, correl[b][None, None, :]], dim=1)

            vid = self.channel_drop(vid)
            vid_masks = vid_masks.unsqueeze(1)
            vid, vid_masks = self.vid_map(vid, vid_masks)


            vid_tmp, vid_masks_tmp = self.fusion(vid, vid_masks, text, text_masks, text_size)
            fpn_tmp, fpn_masks_tmp = self.encode_video(vid_tmp, vid_masks_tmp)
            l, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_masks) # tuple, length=the number of FPN
            fpn_logits_list.append(l)
            fpn_offsets_list.append(o)
            fpn_masks_list.append(m)

        return fpn_logits_list, fpn_offsets_list, fpn_masks_list

    def _drop_forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        assert mv_data is None


        N = self.opt.model.sn
        ratio = self.opt.model.sratio

        if (not eval) and text_size is not None: # and text_size.size(0) != vid.size(0):
            vid = vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_size, dim=0) # nbatch x ntext, nframe
        if eval and (text_size is None) and (text_cls.size(0) != vid.size(0)):
            vid = vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe
        
        if self.opt.model.norm:
            v = shallow_vid / (shallow_vid.norm(dim=1, keepdim=True) + 1e-4)
            t = text_cls / (text_cls.norm(dim=1, keepdim=True) + 1e-4)
            correl = torch.einsum('bht,bh->bt', v, t)  # b x num_text, num_frame
        else:
            correl = torch.einsum('bht,bh->bt', shallow_vid, text_cls)  # b x num_text, num_frame
        all_weight = torch.zeros_like(correl)
        for b in range(correl.shape[0]):
            vid_len = vid_masks[b].sum()
            correl_batch = torch.nn.functional.avg_pool1d(correl[b, None, :vid_len], kernel_size=N, stride=N, ceil_mode=True)[0]
            # correl_batch = torch.nn.functional.adaptive_avg_pool1d(correl[b, None, :vid_len], 16)[0]
            ranked = correl_batch.argsort()
            topk = ranked[-int(ratio * correl_batch.shape[0]):]

            weight = torch.zeros_like(correl_batch)
            weight[topk] = 1
            weight = torch.nn.functional.interpolate(weight[None, None, :], size=vid_len, mode='nearest')[0, 0]
            all_weight[b, :vid_len] = weight
        
        vid = vid * all_weight.unsqueeze(1)
        if not self.opt.model.msf:
            vid_masks = torch.logical_and(all_weight.bool(), vid_masks)
        else:
            vid = torch.cat([vid, shallow_vid], dim=1)
        if self.opt.model.scat:
            vid = torch.cat([vid, correl.unsqueeze(1)], dim=1)

        vid = self.channel_drop(vid)
        vid_masks = vid_masks.unsqueeze(1)
        vid, vid_masks = self.vid_map(vid, vid_masks)

        if not eval:
            if text.ndim == 4:
                text = torch.cat([t[:k] for t, k in zip(text, text_size)])
            if text_masks.ndim == 3:
                text_masks = torch.cat(
                    [t[:k] for t, k in zip(text_masks, text_size)]
                )
            
            text, text_masks = self.encode_text(text, text_masks)
            vid, vid_masks = self.fusion(vid, vid_masks, text, text_masks, text_size)
            fpn, fpn_masks = self.encode_video(vid, vid_masks)
            fpn_logits, fpn_offsets, fpn_masks = \
                self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

            return fpn_logits, fpn_offsets, fpn_masks

        else:
            text_list = text
            text_mask_list = text_masks
            fpn_logits_list = []
            fpn_offsets_list = []
            fpn_masks_list = []
            # if self.opt.model.sratio < 1:
            for i, (text, text_mask) in enumerate(zip(text_list, text_mask_list)):
                vid_tmp = vid[i:i+1]
                vid_masks_tmp = vid_masks[i:i+1] 
                vid_tmp, vid_masks_tmp = self.fusion(vid_tmp, vid_masks_tmp, text, text_mask, text_size)
                fpn_tmp, fpn_masks_tmp = self.encode_video(vid_tmp, vid_masks_tmp)
                l, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_mask) # tuple, length=the number of FPN
                fpn_logits_list.append(l)
                fpn_offsets_list.append(o)
                fpn_masks_list.append(m)
            return fpn_logits_list, fpn_offsets_list, fpn_masks_list


# def expand_frame_label(label, target_len: int):
#     if len(label) == target_len:
#         return label

#     import torch
#     # is_numpy = isinstance(label, np.ndarray)
#     # if is_numpy:
#     #     label = torch.from_numpy(label).float()
#     # if isinstance(label, list):
#     #     label = torch.FloatTensor(label)

#     # label = label.view([1, 1, -1])
#     resized = torch.nn.functional.interpolate(
#         label, size=target_len, mode="nearest"
#     )
#     resized = resized.long()
    
#     # if is_numpy:
#     #     resized = resized.detach().numpy()
#     return resized

class PtTransformerEarlyFusionIterative(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt, second_fusion=True):
        super().__init__()

        self.channel_drop = nn.Dropout1d(opt.model.vid_net.cdrop)
        
        self.opt = opt
        self.text_net = make_text_net(opt.model['text_net'])

        # _opt = opt.model.vid_net.clone()
        in_dim = opt.model.vid_net.in_dim
        if opt.model.msf:
            in_dim *= 2
        if opt.model.scat:
            in_dim += 1
        # self.vid_map = nn.Linear(in_dim, opt.model.vid_net.in_dim)
        self.vid_map = MaskedConv1D(in_dim, opt.model.vid_net.embd_dim, 1)
        _opt = opt.model.vid_net.clone()
        _opt.in_dim = _opt.embd_dim
        self.vid_net = make_video_net(_opt)

        # fusion and prediction heads
        self.fusion = make_fusion(opt.model['fusion'])
        self.cls_head  = make_head(opt.model['cls_head'])
        self.refine = TCN(self.opt.model.vid_net.arch[-1], 32, 32, 
                            num_layers=self.opt.model.vid_net.arch[-1], in_map=True)
        opt.model.cls_head.embd_dim += 32
        self.cls_head2 = make_head(opt.model['cls_head'])
        opt.model.reg_head.embd_dim += 32 
        self.reg_head = make_head(opt.model['reg_head'])


        self.second_fusion = second_fusion

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        if self.second_fusion:
            fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        
        # expand
        # torch.use_deterministic_algorithms(False)
        expand_fpn_logits = [fpn_logits[0]]
        ref_len = fpn_logits[0].shape[1]
        for i, l in enumerate(fpn_logits[1:]):
            new_l = torch.nn.functional.interpolate(l.unsqueeze(1), size=ref_len, mode='nearest')[:, 0]
            new_l = new_l * fpn_masks[0][:, 0]
            expand_fpn_logits.append(new_l)
        expand_fpn_logits = torch.stack(expand_fpn_logits, dim=1) 
        # torch.use_deterministic_algorithms(True)

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

    def forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        if eval:
            return self._drop_forward_eval(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)

        return self._drop_forward(vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size, mv_data, eval)


    def _drop_forward_eval(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        assert mv_data is None
        assert eval


        N = self.opt.model.sn
        ratio = self.opt.model.sratio

        # if (not eval) and text_size is not None: # and text_size.size(0) != vid.size(0):
        #     vid = vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
        #     shallow_vid = shallow_vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
        #     vid_masks = vid_masks.repeat_interleave(text_size, dim=0) # nbatch x ntext, nframe
        assert vid.size(0) == 1, vid.size()
        # if eval and (text_size is None) and (text_cls.size(0) != vid.size(0)):
        #     vid = vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
        #     shallow_vid = shallow_vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
        if self.opt.model.norm:
            v = shallow_vid / (shallow_vid.norm(dim=1, keepdim=True) + 1e-4)
            t = text_cls / (text_cls.norm(dim=1, keepdim=True) + 1e-4)
            correl = torch.einsum('bht,bh->bt', v, t)  # b x num_text, num_frame
        else:
            correl = torch.einsum('bht,bh->bt', shallow_vid, text_cls)  # b x num_text, num_frame

        # vname = self._vname
        # _f, _t = torch.load('/mnt/zijia/long-grounding/exps/clip/results/{}.pt'.format(vname))
        # _f = _f.to('cuda')
        # _t = _t.to('cuda')
        # correl = torch.matmul(_f, _t.T).T # n_query x n_frame
        # assert len(text) == _t.shape[0]
        # assert vid.shape[0] == 1
        # correl = nn.functional.interpolate(correl.unsqueeze(1), size=vid_masks.sum(), mode='linear').squeeze(1)
        # correl = torch.nn.functional.pad(correl, (0, vid_masks.shape[1] - correl.shape[1]), value=0)


        vid_origin = vid
        vid_masks_origin = vid_masks

        text_list = text
        text_mask_list = text_masks
        fpn_logits_list = []
        fpn_offsets_list = []
        fpn_masks_list = []       #     vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe
        for b, (text, text_masks) in enumerate(zip(text_list, text_mask_list)):
            vid = vid_origin.clone()
            vid_masks = vid_masks_origin.clone()
        
            # all_weight = torch.zeros_like(correl)
            vid_len = vid_masks.sum()
            correl_batch = torch.nn.functional.avg_pool1d(correl[b, None, :vid_len], kernel_size=N, stride=N, ceil_mode=True)[0]
            # correl_batch = torch.nn.functional.adaptive_avg_pool1d(correl[b, None, :vid_len], 16)[0]
            ranked = correl_batch.argsort()
            topk = ranked[-int(ratio * correl_batch.shape[0]):]

            weight = torch.zeros_like(correl_batch)
            weight[topk] = 1
            weight = torch.nn.functional.interpolate(weight[None, None, :], size=vid_len, mode='nearest')[0, 0]
            all_weight = torch.zeros_like(vid_masks)
            all_weight[0, :vid_len] = weight
            
            vid = vid * all_weight.unsqueeze(1)
            if not self.opt.model.msf:
                vid_masks = torch.logical_and(all_weight.bool(), vid_masks)
            elif self.opt.model.sfonly:
                vid = shallow_vid
            else:
                vid = torch.cat([vid, shallow_vid], dim=1)
            if self.opt.model.scat:
                vid = torch.cat([vid, correl[b][None, None, :]], dim=1)

            vid = self.channel_drop(vid)
            vid_masks = vid_masks.unsqueeze(1)
            vid, vid_masks = self.vid_map(vid, vid_masks)


            vid_tmp, vid_masks_tmp = self.fusion(vid, vid_masks, text, text_masks, text_size)
            fpn_tmp, fpn_masks_tmp = self.encode_video(vid_tmp, vid_masks_tmp)
            l1, l2, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_masks) # tuple, length=the number of FPN
            fpn_logits_list.append(l2)
            fpn_offsets_list.append(o)
            fpn_masks_list.append(m)

        return fpn_logits_list, fpn_offsets_list, fpn_masks_list

    def _drop_forward(self, vid, shallow_vid, vid_masks, text, text_cls, text_masks, text_size=None, mv_data=None, eval=False):
        """
        vid: b, h, t
        vid_masks: b, t
        """
        assert mv_data is None


        N = self.opt.model.sn
        ratio = self.opt.model.sratio

        if (not eval) and text_size is not None: # and text_size.size(0) != vid.size(0):
            vid = vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_size, dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_size, dim=0) # nbatch x ntext, nframe
        if eval and (text_size is None) and (text_cls.size(0) != vid.size(0)):
            vid = vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            shallow_vid = shallow_vid.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, d, nframe
            vid_masks = vid_masks.repeat_interleave(text_cls.size(0), dim=0) # nbatch x ntext, nframe
        
        if self.opt.model.norm:
            v = shallow_vid / (shallow_vid.norm(dim=1, keepdim=True) + 1e-4)
            t = text_cls / (text_cls.norm(dim=1, keepdim=True) + 1e-4)
            correl = torch.einsum('bht,bh->bt', v, t)  # b x num_text, num_frame
        else:
            correl = torch.einsum('bht,bh->bt', shallow_vid, text_cls)  # b x num_text, num_frame
        all_weight = torch.zeros_like(correl)
        for b in range(correl.shape[0]):
            vid_len = vid_masks[b].sum()
            correl_batch = torch.nn.functional.avg_pool1d(correl[b, None, :vid_len], kernel_size=N, stride=N, ceil_mode=True)[0]
            # correl_batch = torch.nn.functional.adaptive_avg_pool1d(correl[b, None, :vid_len], 16)[0]
            ranked = correl_batch.argsort()
            topk = ranked[-int(ratio * correl_batch.shape[0]):]

            weight = torch.zeros_like(correl_batch)
            weight[topk] = 1
            weight = torch.nn.functional.interpolate(weight[None, None, :], size=vid_len, mode='nearest')[0, 0]
            all_weight[b, :vid_len] = weight
        
        vid = vid * all_weight.unsqueeze(1)
        if not self.opt.model.msf:
            vid_masks = torch.logical_and(all_weight.bool(), vid_masks)
        else:
            vid = torch.cat([vid, shallow_vid], dim=1)
        if self.opt.model.scat:
            vid = torch.cat([vid, correl.unsqueeze(1)], dim=1)

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

        # return fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks
        return fpn_logits1, fpn_logits2, fpn_offsets, fpn_masks

        # else:
        #     text_list = text
        #     text_mask_list = text_masks
        #     fpn_logits_list = []
        #     fpn_offsets_list = []
        #     fpn_masks_list = []
        #     # if self.opt.model.sratio < 1:
        #     for i, (text, text_mask) in enumerate(zip(text_list, text_mask_list)):
        #         vid_tmp = vid[i:i+1]
        #         vid_masks_tmp = vid_masks[i:i+1] 
        #         vid_tmp, vid_masks_tmp = self.fusion(vid_tmp, vid_masks_tmp, text, text_mask, text_size)
        #         fpn_tmp, fpn_masks_tmp = self.encode_video(vid_tmp, vid_masks_tmp)
        #         l, o, m = self.fuse_and_predict(fpn_tmp, fpn_masks_tmp, text, text_mask) # tuple, length=the number of FPN
        #         fpn_logits_list.append(l)
        #         fpn_offsets_list.append(o)
        #         fpn_masks_list.append(m)
        #     return fpn_logits_list, fpn_offsets_list, fpn_masks_list


class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset: # False
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points