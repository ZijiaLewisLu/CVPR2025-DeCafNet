from copy import deepcopy
import math
import yaml
from yacs.config import CfgNode

CN = CfgNode()

CN.seed = None # 1234567891

CN.task = 'snag'

CN.aux = aux =  CfgNode()
aux.gpu = None
aux.mark = "" # for adding addtional note
aux.runid = 0 # the X-th run of this configuration
aux.debug = False
aux.wandb_project = "mv-nlq"
aux.wandb_id = None
aux.log = "snag"
# aux.wandb_offline = False
aux.resume = True # "", ckpt_path, "max" (resume latest ckpt of the experiment)
aux.model_resume = None # only load model weight, do not load other states
aux.distributed = False
aux.eval_run = -1 # False
aux.eval_by = 'epoch'
aux.save_run = 2
aux.log_interval = 100
aux.mount = '/home/azureuser/local/snag'
aux.is_submit = False
aux.region = 'scus'
aux.dryrun = False
aux.ntd = None
aux.pretrain = None
aux.cache_size = 600000

# aux.ngpu = 1
# aux.continue_from = None
aux.tag = None

aux.extract_feature = False
# aux.

def base_data():    
    CN.data = data = CfgNode()
    data.name = 'video_centric'
    data.split = 'train'
    data.eval_split = 'val'
    data.anno_file = None 
    data.vid_feat_dir = None #./data/ego4d/egovlp_features/video
    data.vid_load = 'npy'
    data.shallow_vid_feat_dir = None #./data/ego4d/egovlp_features/video
    data.shallow_vid_load = 'npy'
    data.shallow_ds = None
    data.text_feat_dir = None
    data.ext_score_dir = None

    data.text_cls_fname = None
    data.clip_token_fname = None

    data.ego4d_train_anno = None
    data.ego4d_val_anno = None
    data.ego4d_metadata   = None
    data.video_dir        = None

    data.clip_size = 32
    data.clip_stride = 8
    data.to_fixed_len = False
    data.downsample_rate = 1
    data.true_ds = None
    data.max_num_text = 2
    data.trunc_thresh = 0.5
    data.crop_ratio = (0.9, 1.0)


def grounder():

    CN.model = model = CfgNode()
    model.name = 'default'

    model.text_net = CfgNode()
    model.text_net.name = 'transformer'
    model.text_net.in_dim = 300
    model.text_net.embd_dim = 128
    model.text_net.max_seq_len = 24
    model.text_net.n_heads = 4
    model.text_net.use_abs_pe = False
    model.text_net.use_bkgd_token = True

    model.vid_net = CfgNode()
    model.vid_net.name = 'transformer'
    model.vid_net.in_dim = 500
    model.vid_net.embd_dim = 128
    model.vid_net.n_heads = 4
    model.vid_net.max_seq_len = 256
    model.vid_net.stride = 1
    model.vid_net.arch = (2, 0, 7)
    model.vid_net.mha_win_size = 5
    model.vid_net.attn_pdrop = 0.0
    model.vid_net.proj_pdrop = 0.1
    model.vid_net.path_pdrop = 0.1
    model.vid_net.use_abs_pe = True
    model.vid_net.fuse = 'cat'
    model.vid_net.pool_only = False
    model.vid_net.cdrop = 0.0

    model.fusion = CfgNode()
    model.fusion.name = 'xattn'
    model.fusion.n_layers = 2
    model.fusion.n_heads = 4
    model.fusion.attn_pdrop = 0.0
    model.fusion.proj_pdrop = 0.1
    model.fusion.path_pdrop = 0.1
    model.fusion.xattn_mode = 'adaln'

    model.cls_head = CfgNode()
    model.cls_head.name = 'cls'
    model.cls_head.n_layers = 2
    model.cls_head.prior_prob = 0.0

    model.reg_head = CfgNode()
    model.reg_head.name = 'reg'
    model.reg_head.n_layers = 2

    model.pretrain = None
    model.sratio = 0.0
    model.sn = 60
    model.msf = False # merge shallow and full features
    model.scat = False
    model.sfonly = False
    model.norm = False

    loss = CN.loss = CfgNode()
    loss.fc_a = 0.5 # alpha for focal loss
    loss.fc_s = 0.2 # label smoothing for focal loss

    # infuse = CN.infuse = CfgNode()
    # infuse.name = None
    # infuse.layers = 1
    # infuse.n_heads = 0
    # infuse.drgb = 0.0

    CN.pt_gen = pt_gen = CfgNode()
    pt_gen.regression_range = 4
    pt_gen.sigma = 0.5

    CN.train = train = CfgNode()
    train.batch_size = 16
    train.num_workers = 4
    train.epochs = 5
    train.warmup_epochs = 5
    train.ema_beta = 0.999
    train.center_sampling = 'radius'
    train.center_sampling_radius = 1.5
    train.loss_norm = 160
    train.loss_norm_momentum = 0.9
    train.loss_weight = 1.0
    train.reg_loss = 'diou'
    train.version = 1

    CN.optimizer = optimizer = CfgNode()
    optimizer.name = 'adamw'
    optimizer.lr = 1e-3
    optimizer.weight_decay = 0.05
    optimizer.clip_grad_norm = 1.0

    CN.scheduler = scheduler = CfgNode()
    scheduler.name = 'multistep'
    scheduler.steps = (-1,)
    scheduler.gamma = 0.1


    base_data()

    CN.eval = eval = CfgNode()
    # eval.data = data = CfgNode()
    # data.split = 'test'
    # data.name = 'video_centric'
    eval.ranks = (1, 5)
    eval.iou_threshs = (0.3, 0.5)
    eval.pre_nms_thresh = 0.001
    eval.pre_nms_topk = 2000
    eval.seg_len_thresh = 0.1

    eval.data = CN.data.clone()
    for k in eval.data:
        eval.data[k] = None

    CN.nms = nms = CfgNode()
    nms.mode = 'soft_nms'
    nms.iou_thresh = 0.1
    nms.min_score = 0.001
    nms.max_num_segs = 5
    nms.sigma = 0.9
    nms.voting_thresh = 0.95

    CN.log = log = CfgNode()
    log.log_interval = 100
    log.checkpoint_epochs = (6, 7, 8, 9, 10)

    CN.aux.download_mv_feat = False

def mad():

    CN.model = model = CfgNode()
    model.name = 'default'

    model.text_net = CfgNode()
    model.text_net.name = 'transformer'
    model.text_net.in_dim = 300
    model.text_net.embd_dim = 128
    model.text_net.max_seq_len = 24
    model.text_net.n_heads = 4
    model.text_net.use_abs_pe = False
    model.text_net.use_bkgd_token = True

    model.vid_net = CfgNode()
    model.vid_net.name = 'transformer'
    model.vid_net.in_dim = 500
    model.vid_net.embd_dim = 128
    model.vid_net.n_heads = 4
    model.vid_net.max_seq_len = 256
    model.vid_net.stride = 1
    model.vid_net.arch = (2, 0, 7)
    model.vid_net.mha_win_size = 5
    model.vid_net.attn_pdrop = 0.0
    model.vid_net.proj_pdrop = 0.1
    model.vid_net.path_pdrop = 0.1
    model.vid_net.use_abs_pe = True
    model.vid_net.fuse = 'cat'
    model.vid_net.pool_only = False
    model.vid_net.cdrop = 0.0

    model.fusion = CfgNode()
    model.fusion.name = 'xattn'
    model.fusion.n_layers = 2
    model.fusion.n_heads = 4
    model.fusion.attn_pdrop = 0.0
    model.fusion.proj_pdrop = 0.1
    model.fusion.path_pdrop = 0.1
    model.fusion.xattn_mode = 'adaln'

    model.cls_head = CfgNode()
    model.cls_head.name = 'cls'
    model.cls_head.n_layers = 2
    model.cls_head.prior_prob = 0.0

    model.reg_head = CfgNode()
    model.reg_head.name = 'reg'
    model.reg_head.n_layers = 2

    model.pretrain = None
    model.norm = False

    loss = CN.loss = CfgNode()
    loss.fc_a = 0.5 # alpha for focal loss
    loss.fc_s = 0.2 # label smoothing for focal loss

    CN.pt_gen = pt_gen = CfgNode()
    pt_gen.regression_range = 4
    pt_gen.sigma = 0.5

    CN.train = train = CfgNode()
    train.batch_size = 16
    train.num_workers = 4
    train.epochs = 5
    train.warmup_epochs = 5
    train.ema_beta = 0.999
    train.center_sampling = 'radius'
    train.center_sampling_radius = 1.5
    train.loss_norm = 160
    train.loss_norm_momentum = 0.9
    train.loss_weight = 1.0
    train.reg_loss = 'diou'
    train.version = 1
    train.microbatch_size = 1

    CN.optimizer = optimizer = CfgNode()
    optimizer.name = 'adamw'
    optimizer.lr = 1e-3
    optimizer.weight_decay = 0.05
    optimizer.clip_grad_norm = 1.0

    CN.scheduler = scheduler = CfgNode()
    scheduler.name = 'multistep'
    scheduler.steps = (-1,)
    scheduler.gamma = 0.1


    base_data()

    CN.eval = eval = CfgNode()
    # eval.data = data = CfgNode()
    # data.split = 'test'
    # data.name = 'video_centric'
    eval.ranks = (1, 5)
    eval.iou_threshs = (0.3, 0.5)
    eval.pre_nms_thresh = 0.001
    eval.pre_nms_topk = 2000
    eval.seg_len_thresh = 0.1
    eval.max_vid_len = 32768

    eval.data = CN.data.clone()
    for k in eval.data:
        eval.data[k] = None

    CN.nms = nms = CfgNode()
    nms.mode = 'soft_nms'
    nms.iou_thresh = 0.1
    nms.min_score = 0.001
    nms.max_num_segs = 5
    nms.sigma = 0.9
    nms.voting_thresh = 0.95

    CN.log = log = CfgNode()
    log.log_interval = 100
    log.checkpoint_epochs = (6, 7, 8, 9, 10)

    # CN.aux.download_mv_feat = False

# def egovlp():

#     CN.model = model = CfgNode()
#     model.patch_size = 16
#     model.vid_dim = 256
#     model.text_dim = 256
#     model.dim = 768
#     model.depth = 12
#     model.n_heads = 12

#     model.pretrain = None

#     model.tau = 1
#     model.gc_seg = None

    # CN.mv_enc = mv_enc = CfgNode()
    # mv_enc.name = None

    # mv_enc.in_size = 224
    # mv_enc.in_stride = 3

    # mv_enc.s_layers = 2
    # mv_enc.t_layers = 2
    # mv_enc.n_heads = 8
    # mv_enc.patch_size = 16
    # mv_enc.dim = 128

    # mv_enc.pe = 'sin'

    # mv_enc.load = None
    # mv_enc.finetune = False

    # mv_enc.depth = None
    # mv_enc.head = 'avg'
    
    # mv_enc.pretrain = None
    # mv_enc.pre_type = 1 # 1: timesformer, 2: our model

    # mv_enc.sep_patch = False

    # mv_enc.md = False # merge source frame direction
    base_data()
    data = CN.data
    data.clip_d = 4.0 # 16 / 4 
    data.small_img_size = 112
    data.max_feats = 16
    data.npos_feats = 8

    loss = CN.loss = CfgNode()
    loss.dw = 0.0

    optimizer = CN.optimizer
    optimizer.fpretrain = False
    optimizer.lpretrain = None


    # data = CN.data
    # data.tanh_max = None
    # data.md = False # merge source frame direction

    # mv_enc.dec_layers = 4
    # mv_enc.dec_dim = 256
    # mv_enc.dec_nheads = 8

    # mv_enc.pt_cw = 8
    # mv_enc.pt_dp = 'rnd'
    # mv_enc.pt_dpr = 0.5
    # mv_enc.vlw = 0.1

# def end2end():

#     grounder()

#     CN.model.on = False
#     del CN['infuse']
#     CN.model.loss = CN.loss
#     del CN['loss']

#     # parameter for 
#     CN.encoder = encoder = CfgNode()
#     encoder.patch_size = 16
#     encoder.vid_dim = 256
#     encoder.text_dim = 256
#     encoder.dim = 768
#     encoder.depth = 12
#     encoder.n_heads = 12
#     encoder.drate = 0.0 # dropout rate
#     encoder.dprate = 0.1 # drop path rate
#     encoder.tau = 1
#     encoder.output = 'distill'

#     encoder.pretrain = None
#     encoder.resume = None

#     encoder.gc_seg = None

#     encoder.dw = 0.0
#     encoder.cw = 1.0
#     encoder.clv = 1 # version of contrastive loss

#     encoder.itp = False

#     encoder.sp_loc = None # 'patch'
#     encoder.tp_loc = None
#     encoder.ptype = 'conv'

#     encoder.ap_loc = None
#     encoder.ap_layer = 2
#     encoder.ap_dim = 256

#     encoder.fd = 1 # encoder feature dilation

#     data = CN.data
#     data.clip_d = 4.0 # 16 / 4 
#     data.img_size = 224
#     del data['shallow_vid_feat_dir'] 
#     del data['shallow_vid_load'] 

#     # data.max_feats = 16
#     # data.npos_feats = 8

#     # loss = CN.loss = CfgNode()
#     # loss.dw = 0.0

#     optimizer = CN.optimizer
#     # optimizer.fpretrain = False
#     optimizer.lpretrain = None
#     optimizer.lenc = 1e-4

#     CN.train.version = 1

#     eval = CN.eval 
#     eval.data = CN.data.clone()
#     for k in eval.data:
#         eval.data[k] = None



def _update_opt(opt, is_training=True):
    max_text_len = opt['model']['max_text_len'] = opt['model']['text_net']['max_seq_len']
    max_vid_len = opt['model']['max_vid_len'] = opt['model']['vid_net']['max_seq_len']
    vid_stride = opt['model']['vid_stride'] = opt['model']['vid_net']['stride']
    # num_fpn_levels = opt['model']['vid_net']['arch'][-1]
    num_fpn_levels = opt['model']['num_fpn_levels'] = opt['model']['vid_net']['arch'][-1]
    opt['model']['mha_win_size'] = opt['model']['vid_net']['mha_win_size']
    opt['data']['max_text_len'] = max_text_len
    opt['data']['max_vid_len'] = vid_stride * max_vid_len
    opt['scheduler']['epochs'] = opt['train']['epochs']
    opt['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']

    # eval_data_opt = deepcopy(opt['train']['data'])
    # eval_data_opt['name'] = opt['eval']['data']['name']
    # eval_data_opt['split'] = opt['eval']['data']['split']
    # opt['eval']['data'] = eval_data_opt

    text_dim = opt['model']['text_net']['embd_dim']
    vid_dim = opt['model']['vid_net']['embd_dim']
    opt['model']['fusion']['text_dim'] = text_dim
    opt['model']['fusion']['vid_dim'] = vid_dim
    opt['model']['cls_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['num_fpn_levels'] = num_fpn_levels
    # n = 1
    # if not is_training:
    #     n = math.ceil(
    #         opt['eval'].get('max_vid_len', max_vid_len * 4) / max_vid_len
    #     )
    opt['pt_gen']['num_fpn_levels'] = num_fpn_levels
    opt['pt_gen']['max_seq_len'] = max_vid_len * 4

    # if opt.infuse.name is None and opt.model.vid_net.fuse is not None:
    #     opt.infuse.name = opt.model.vid_net.fuse
    

def _update_eval_data(opt):
    for k in opt.eval.data:
        if opt.eval.data[k] is None:
            opt.eval.data[k] = opt.data[k]
        if opt.data.eval_split is not None:
            opt.eval.data['split'] = opt.data.eval_split

def short_video_grounder():
    CN.data.group_method = None
    CN.eval.data.group_method = None
    CN.data.tokenizer = None
    CN.eval.data.tokenizer = None

    # multi-stage temporal refinement
    CN.model.rlayer = None
    CN.model.rdim = 32 




def get_cfg_defaults(task=None):
    if task == 'grounder':
        grounder()
    elif task == 'mad':
        mad()
    elif task == 'charades':
        mad()
        short_video_grounder()

    # elif task == 'saliency':
    #     snag()
    #     egovlp()

    #     CN.train.version = 2 # trainer version 

    #     CN.eval = eval = CfgNode()
    #     eval.data = CN.data.clone()
    #     for k in eval.data:
    #         eval.data[k] = None

    elif task == 'end2end':
        # snag()
        end2end()

        # CN.eval = eval = CfgNode()
        # eval.data = CN.data.clone()
        # for k in eval.data:
        #     eval.data[k] = None

    # elif task == 'c3d':
    #     end2end()
    #     CN.data.tokenizer = None

    return CN.clone()