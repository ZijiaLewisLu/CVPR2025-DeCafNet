from yacs.config import CfgNode
from ..core.opt import _update_opt, get_cfg_defaults, _update_eval_data
import os
import random 
from .dist_utils import get_rank, barrier, get_local_rank
import torch
import numpy as np
import pickle
import logging
import os
import json
import numpy as np

def random_seed(length):
    random.seed()
    min = 10**(length-1)
    max = 9*min + (min-1)
    return random.randint(0, max)

def _cfg2flatdict_helper(cfg: CfgNode) -> dict:
    D = {}
    for k, v in cfg.items():
        if not isinstance(v, CfgNode):
            D[k] = v
        else:
            d = _cfg2flatdict_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)
            
    return D

def type_convert_helper(x):
    import torch
    t = type(x)
    if t in [int, float, bool, str, torch.Tensor]:
        return x
    else:
        return str(x)

def cfg2flatdict(cfg: CfgNode, type_convert=True) -> dict:
    D = {}
    for k, v in cfg.items():
        if not isinstance(v, CfgNode):
            D[k] = v
        else:
            d = _cfg2flatdict_helper(v)
            d = { "%s.%s" % (k, k2): v for k2, v in d.items() }
            D.update(d)

    if type_convert:
        D = { k:type_convert_helper(v) for k, v in D.items() }
            
    return D


def generate_diff_dict(default: CfgNode, cfg: CfgNode, include_missing=False) -> dict :
    """
    include_missing = False
        if a key is missing in cfg,
        it assumes the value matches with that of default
    """

    diff = {}
    for k, v in cfg.items():
        if k not in default:
            if include_missing:
                print(k)
            continue
        if isinstance(v, CfgNode):
            subdiff = generate_diff_dict(default[k], cfg[k], include_missing=include_missing)
            if len(subdiff) > 0:
                diff[k] = subdiff
        else:
            if v != default[k]:
                diff[k] = v
    
    return diff

def capitalize(string):
    return string[0].upper() + string[1:]

def diff2expname(diff: dict, remove_leaf=False):
    string = ""
    for k, v in diff.items():
        if k.lower()  == "aux":
            continue # exclude auxiliary config
        if k.lower() == "split":
            continue # exclude split name

        if isinstance(v, dict):
            v = diff2expname(v, remove_leaf=False) # when recursive call, always false
            string += "%s[%s]_" % (k, v)
        elif not remove_leaf:
            if isinstance(v, bool):
                v = str(v)[0]
            string += "%s:%s_" % (k, v)
    
    string = string[:-1] # remove last dash
    return string


_CONFIG_FILE_DICT = {}

def generate_expname(cfg:CfgNode, cfg_file=None, default=None) -> str:
    if cfg_file is None:
        cfg_file = cfg.aux.cfg_file

    expname = []

    # add cfg_file and generate reference cfg
    if default is None:
        default = get_cfg_defaults()
    else:
        default = default.clone()

    for f in cfg_file:
        if f not in _CONFIG_FILE_DICT:
            with open(f, 'r') as fp:
                _CONFIG_FILE_DICT[f] = CfgNode.load_cfg(fp)

        default.merge_from_other_cfg(_CONFIG_FILE_DICT[f])

        f = os.path.basename(f)
        f = '.'.join(f.split('.')[:-1])
        expname.append(f)


    # add other setting
    diff = generate_diff_dict(default, cfg)
    prune = {}
    for k, v in diff.items():
        prune[capitalize(k)] = v
    diff_string = diff2expname(prune)
    if len(diff_string) > 0:
        expname.append(diff_string)
    if len(cfg.aux.mark) > 0:
        expname.append(cfg.aux.mark)

    expname = '_'.join(expname)
    return expname


def int2float_check(x, tgt):
    if isinstance(tgt, float) and "." not in x:
        try:
            int(x) # first check if x can convert to int
            x = x + '.0' # if can convert, change to float match str
        except ValueError:
            pass # cannot convert, pass on to cfg to throw error
    return x

def hiedict2cfg(cfg_dict:dict) -> CfgNode:
    cfg = CfgNode()
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            v = hiedict2cfg(v)
        cfg[k] = v
    return cfg

def _get_var(c, ks: list, delete=False):
    if len(ks) == 1:
        assert ks[0] in c, (ks[0], c)
        v = c[ks[0]]
        if delete:
            del c[ks[0]]
        return v
    else:
        return _get_var(c[ks[0]], ks[1:], delete=delete)



def get_task(cfg_file=[], set_cfgs=None):
    key = 'task'
    task = get_cfg_defaults().task

    for fname in cfg_file:
        with open(fname, "r") as f:
            cfg = CfgNode().load_cfg(f)
        task = cfg.get(key, task)
    
    L = len(set_cfgs) if set_cfgs else 0
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if k == key:
            task = v

    return task




def setup_cfg(cfg_file=[], set_cfgs=None, default: CfgNode=None, logdir="log/") -> CfgNode:
    """
    update default cfg according to cmd line input
    and automatic generate experiment name
    """

    task = get_task(cfg_file, set_cfgs)
    assert task is not None, (task, cfg_file, set_cfgs)
    print('task', task)

    if default is None:
        cfg = get_cfg_defaults(task)
    else:
        cfg = default.clone()

    # preprocess set_cfgs to convert int2float
    L = len(set_cfgs) if set_cfgs else 0
    new_set_cfgs = []
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if not isinstance(k, list):
            k = [k]
        for k_ in k:
            tgt = _get_var(cfg, k_.split('.'))
            v_ = int2float_check(v, tgt)
            new_set_cfgs.extend([k_, v_])


    # update cfg
    for f in cfg_file: # if no config file, this is empty list
        cfg.merge_from_file(f)
    if set_cfgs is not None:
        cfg.merge_from_list(new_set_cfgs)

    cfg.aux.cfg_file = cfg_file
    cfg.aux.set_cfgs = set_cfgs

    # generate experiment name
    cfg.aux.exp = generate_expname(cfg, default=default)


    # create name of logdir
    if cfg.aux.debug:
        logdir = os.path.join('log', 'test', cfg.aux.exp, str(cfg.aux.runid))
    else:
        logdir = os.path.join('log', cfg.aux.log, cfg.aux.exp, str(cfg.aux.runid))

    barrier()

    # resume random seed and wandb
    prev_opt = os.path.join(logdir, 'opt.yaml')
    if os.path.exists(prev_opt):
        with open(prev_opt, 'r') as fp:
            prev_opt = CfgNode.load_cfg(fp)

        if prev_opt.aux.get('wandb_id', None) is not None:
            cfg.aux.wandb_id = prev_opt.aux.get('wandb_id', None)
            print('Resume Wandb ID:', cfg.aux.wandb_id)

        if cfg.seed is None and prev_opt.seed is not None:
            cfg.seed = prev_opt.seed
            print('Resume Random Seed:', cfg.seed)

    if cfg.seed is None:
        cfg.seed = random_seed(9)
        print('Use Random Seed:', cfg.seed)

    if task in ['ego4d', 'encoder', 'mad', 'svg']:
        _update_opt(cfg)

    _update_eval_data(cfg)

    cfg.aux.logdir = logdir
    return cfg


def update_from(cfg: CfgNode, ref: CfgNode, inplace=False) -> CfgNode:
    if not inplace:
        cfg = cfg.clone()
    cfg.defrost()

    for k in ref:
        if k not in cfg:
            cfg[k] = ref[k]
        elif isinstance(cfg[k], CfgNode):
            update_from(cfg[k], ref[k], inplace=True)
    
    return cfg


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x








def count_parameters(model, scale=1e6, trained_only=True):
    if trained_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / scale
    else:
        return sum(p.numel() for p in model.parameters()) / scale

def create_wandb(opt):
    import wandb
    wandb.require("core")

    wandb_kwargs = dict(
        project=opt.aux.wandb_project,
        dir=opt.aux.logdir,
        group=opt.aux.exp,
        id=opt.aux.get('wandb_id', None),
        resume="allow",
        config=cfg2flatdict(opt),
        reinit=True,
        save_code=False,
        mode="offline" if (opt.aux.debug) else "online",
        notes="log_dir: " + opt.aux.logdir,
        job_type=opt.aux.tag,
    )
    if opt.aux.get('wandb_entity', None):
        wandb_kwargs['entity'] = opt.aux.wandb_entity

    run = wandb.init(**wandb_kwargs)
    opt.aux.wandb_id = run.id
    return run






            


    


def easy_reduce(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)

    if len(scores) == 0:
        return np.nan

    elif isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )

    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)

    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )
        average = tuple(average)

    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = easy_reduce([s[k] for s in scores], mode=mode, skip_nan=skip_nan)

    elif isinstance(scores[0], float) or isinstance(scores[0], int) or isinstance(scores[0], np.float32): # TODO - improve
        if skip_nan:
            scores = [ x for x in scores if not np.isnan(x) ]

        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]) )

    return average

def cv_get_video_info(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    # the frame rate or frames per second
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # the total number of frames
    total_num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # the duration in seconds
    if frame_rate == 0:
        duration = 0
    else:
        duration = total_num_frames / frame_rate

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()

    info = {
        'fps': frame_rate,
        'nframes': total_num_frames,
        'duration': duration,
        'width': width,
        'height': height
    }
    return info


def compute_t_for_f(start_f, end_f, clip_size, clip_stride, in_clip_stride, start_t=0, end_t=None):
    clip_offset = int(0.5 * clip_size / clip_stride)

    window = np.arange(0, clip_size, step=in_clip_stride) - (clip_size//2)
    
    fidx = np.arange(start_f, end_f).reshape(-1, 1) # nframes x 1
    fidx2tlist_raw = (fidx + clip_offset) * clip_stride + window # nframes x clip_size


    assert fidx2tlist_raw.min() >= start_t

    if end_t is not None:
        fidx2tlist = np.clip(fidx2tlist_raw, 0, end_t-1)
    else:
        fidx2tlist = fidx2tlist_raw
    fidx2tlist = fidx2tlist - start_t

    tlist = np.unique(fidx2tlist.flatten())
    full_tlist = np.unique(fidx2tlist_raw.flatten())
    for _ in range(len(full_tlist) - len(tlist)):   
        tlist = np.concatenate([tlist, [tlist[-1]]])

    _map = {t: i for i, t in enumerate(tlist)}
    f2t_rel = np.array([_map[x] for x in fidx2tlist.reshape(-1)])
    f2t_rel = f2t_rel.reshape(fidx2tlist.shape)
    f2t_rel = torch.from_numpy(f2t_rel)

    return tlist, f2t_rel

def compute_iou(intervals1, intervals2):
    """
    Compute pairwise Intersection over Union (IoU) between two lists of time intervals.

    :param intervals1: List of intervals [(start1, end1), ...]
    :param intervals2: List of intervals [(start2, end2), ...]
    :return: List of IoU values for each pair of intervals (one from each list)
    """
    def interval_intersection(start1, end1, start2, end2):
        """Calculate the intersection length of two intervals."""
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        return max(0, intersection_end - intersection_start)
    
    def interval_union(start1, end1, start2, end2):
        """Calculate the union length of two intervals."""
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        return union_end - union_start

    iou_results = []

    for (start1, end1) in intervals1:
        for (start2, end2) in intervals2:
            intersection_length = interval_intersection(start1, end1, start2, end2)
            union_length = interval_union(start1, end1, start2, end2)
            iou = intersection_length / union_length if union_length > 0 else 0
            iou_results.append((start1, end1, start2, end2, iou))
    
    return iou_results


def interpolate_array(original_array, target_length, kind='linear'):
    from scipy.interpolate import interp1d
    """
    Interpolates the original array to the target length.

    Parameters:
    - original_array (np.ndarray): The array to be interpolated.
    - target_length (int): The desired length of the interpolated array.
    - kind (str): The type of interpolation ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.)

    Returns:
    - np.ndarray: The interpolated array.
    """
    
    # Ensure the input array is a NumPy array
    original_array = np.asarray(original_array)
    
    # Original indices
    original_indices = np.arange(len(original_array))
    
    # New indices
    new_indices = np.linspace(0, len(original_array) - 1, target_length)
    
    # Create interpolation function
    interp_function = interp1d(original_indices, original_array, kind=kind, fill_value='extrapolate')
    
    # Interpolate and return the result
    interpolated_array = interp_function(new_indices)
    
    return interpolated_array
