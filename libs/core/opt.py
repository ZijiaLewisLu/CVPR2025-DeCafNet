from yacs.config import CfgNode

CN = CfgNode()

CN.seed = None # 1234567891
CN.task = 'ego4d'

CN.data = data = CfgNode()
data.name = 'video_centric'
data.split = 'train'
data.eval_split = 'val'
data.anno_file = None 
data.vid_feat_dir = None #./data/ego4d/egovlp_features/video
data.vid_load = 'npy'
data.sidekick_vid_feat_dir = None #./data/ego4d/egovlp_features/video
data.sidekick_vid_load = 'npy'
data.sidekick_ds = 1
data.text_feat_dir = None
data.ext_score_dir = None

data.text_cls_fname = None
data.clip_token_fname = '/tmp/decafnet/data/datasets/clip/nlq_token_features_B_32.npy'

data.clip_size = 32
data.clip_stride = 8
data.to_fixed_len = False
data.downsample_rate = 1
# data.true_ds = None
data.max_num_text = 2
data.trunc_thresh = 0.5
data.crop_ratio = (0.9, 1.0)


CN.aux = aux = CfgNode()
aux.gpu = None
aux.mark = "" # for adding addtional note
aux.runid = 0 # the X-th run of this configuration
aux.debug = False
aux.wandb_enable = False
aux.wandb_project = "decafnet"
aux.wandb_entity = None
aux.wandb_id = None
aux.log = "snag"
aux.eval_run = -1 # False
aux.eval_by = 'epoch'
# aux.save_run = 2
aux.log_interval = 100
aux.dryrun = False

aux.tag = None



def _init_decafnet_shared_defaults():

    CN.model = model = CfgNode()

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

    # if add_svg_data_fields:
    CN.data.group_method = None
    CN.data.tokenizer = None

    CN.eval = eval = CfgNode()
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

def snag():
    _init_decafnet_shared_defaults()
    CN.task = 'ego4d'
    model = CN.model
    model.sratio = 0.0
    model.sn = 60
    model.msf = False # merge shallow and full features
    model.scat = False
    model.sfonly = False

def mad():
    _init_decafnet_shared_defaults()
    CN.task = 'mad'


def svg():
    _init_decafnet_shared_defaults()
    CN.task = 'svg'

    model = CN.model
    # multi-stage temporal refinement
    model.rlayer = None
    model.rdim = 32
    model.name = None


def _update_opt(opt):
    max_text_len = opt['model']['max_text_len'] = opt['model']['text_net']['max_seq_len']
    max_vid_len = opt['model']['max_vid_len'] = opt['model']['vid_net']['max_seq_len']
    vid_stride = opt['model']['vid_stride'] = opt['model']['vid_net']['stride']
    num_fpn_levels = opt['model']['num_fpn_levels'] = opt['model']['vid_net']['arch'][-1]
    opt['model']['mha_win_size'] = opt['model']['vid_net']['mha_win_size']
    opt['data']['max_text_len'] = max_text_len
    opt['data']['max_vid_len'] = vid_stride * max_vid_len
    opt['scheduler']['epochs'] = opt['train']['epochs']
    opt['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']


    text_dim = opt['model']['text_net']['embd_dim']
    vid_dim = opt['model']['vid_net']['embd_dim']
    opt['model']['fusion']['text_dim'] = text_dim
    opt['model']['fusion']['vid_dim'] = vid_dim
    opt['model']['cls_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['num_fpn_levels'] = num_fpn_levels
    opt['pt_gen']['num_fpn_levels'] = num_fpn_levels
    opt['pt_gen']['max_seq_len'] = max_vid_len * 4

def _update_eval_data(opt):
    for k in opt.data:
        if (k not in opt.eval.data) or (opt.eval.data[k] is None):
            opt.eval.data[k] = opt.data[k]
    if opt.data.eval_split is not None:
        opt.eval.data.split = opt.data.eval_split



def get_cfg_defaults(task=None):
    task = task or CN.task
    if task == 'ego4d':
        snag()
    elif task == 'mad':
        mad()
    elif task == 'svg':
        svg()
    else:
        raise ValueError(f'unsupported task: {task}')

    return CN.clone()
