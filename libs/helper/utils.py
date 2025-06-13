import torch
import numpy as np
import pickle
import logging
import os
import json
import numpy as np

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x

class Video():

    def __init__(self, vname=None):
        self.vname = vname


class Checkpoint():
    """
    for a checkpoint, 
    firstly load out construct a Video object for each video,
    then compute statistics, such as performance,
    """

    __VERSION__ = 1.0
    __DATE__    = "5-13"

    def __init__(self, iteration):

        self.iteration = iteration
        self.videos = {}

        self.__version__ = Checkpoint.__VERSION__
        self.__date__ = Checkpoint.__DATE__

    def add_videos(self, videos: list):
        if isinstance(videos, Video):
            videos = [videos]
        for v in videos:
            self.videos[v.vname] = v

    def drop_videos(self):
        self.videos = {}

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as fp:
            ckpt = pickle.load(fp)
            if not ckpt.__version__ == Checkpoint.__VERSION__:
                logging.warning("old version checkpoint found %s" % ckpt.__version__)
                logging.warning(fname)
        return ckpt
    
    def save(self, fname):
        self.fname = fname
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def _random_video(self):
        vnames = list(self.videos.keys())
        vname = np.random.choice(vnames, 1).item()
        return vname, self.videos[vname]

    def clean_attr(self, video_attr):
        for vname, video in self.videos.items():
            if hasattr(video, video_attr):
                delattr(video, video_attr)

    def compute_iou(self):
        for vname, video in self.videos.items():
            video.top1_ious = []
            video.ious = []
            for i in range(len(video.abs_target)):
                ious = compute_iou(video.abs_target[i:i+1], video.results[i]['segments'].detach().cpu().numpy())
                if len(ious) == 0:
                    continue
                ious = [ x[-1] for x in ious ] # ious[0][-1]
                video.ious.append(ious)
                video.top1_ious.append(ious[0])

            video.top1_ious = np.array(video.top1_ious)
            video.ious = np.array(video.ious)


def count_parameters(model, scale=1e6, trained_only=True):
    if trained_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / scale
    else:
        return sum(p.numel() for p in model.parameters()) / scale

def create_wandb(opt):
    from libs.core.utils import cfg2flatdict
    import wandb
    wandb.require("core")
    # run = wandb.init(
    #             project=opt.aux.wandb_project, entity="zijia",
    #             dir=opt.aux.logdir,
    #             group=opt.aux.exp, resume="allow",
    #             config=cfg2flatdict(opt),
    #             reinit=True, save_code=False,
    #             mode="offline" if opt.aux.debug else "online",
    #             )
    # return run

    # wandb_runid = resume_wandb_runid(logdir)
    os.environ['WANDB_API_KEY'] = '1d7e4028d6bca114b3348f1628c0b3fddae70dae'
    run = wandb.init(
                project=opt.aux.wandb_project, entity="zijia",
                dir=opt.aux.logdir,
                group=opt.aux.exp, 
                id=opt.aux.get('wandb_id', None), resume="allow",
                config=cfg2flatdict(opt),
                reinit=True, save_code=False,
                mode="offline" if (opt.aux.debug) else "online",
                notes="log_dir: " + opt.aux.logdir,
                job_type=opt.aux.tag,
                )
    opt.aux.wandb_id = run.id
    return run

def create_monitor_wandb(name, group='track', project='moniter'):
    import wandb
    # import tempfile
    wandb.require("core")
    os.environ['WANDB_API_KEY'] = '1d7e4028d6bca114b3348f1628c0b3fddae70dae'
    run = wandb.init(
                project=project, entity="zijia",
                group=group,
                job_type=group,
                config={'name': name}
                )
    return run


def parse_ego4d_nlq(anno="/tmp/zijia-2024/datasets/ego4d_data/v2/annotations/nlq_val.json"):
    import json

    with open(anno) as fp:
        anno = json.load(fp)

    video_clip_info = {}
    for v in anno['videos']:
        for clip in v['clips']:
            d = {k:v for k, v in clip.items()}
            for k in ['video_start_sec', 'video_end_sec']: # 'video_start_frame', 'video_end_frame']:
                d[k] = clip[k]
            d['video_uid'] = v['video_uid']

            queries = []
            for atr in clip['annotations']:
                queries.extend(atr['language_queries'])
            
            d['annotations'] = queries

            # meta = v2meta[v['video_uid']]
            # d['video_nframes'] = meta['num_frames']
            # h, w = meta['display_resolution_height'], meta['display_resolution_width']
            # if h is None and w is None:
            #     assert v['video_uid'].startswith('grp'), meta
            #     h, w = 1440, 1920
            # d['height'] = h
            # d['width'] = w

            video_clip_info[clip['clip_uid']] = d
    
    return video_clip_info


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

    # start_offset = pi * self.frame_per_partition # 1000
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