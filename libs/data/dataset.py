from collections import OrderedDict
from copy import deepcopy
from functools import partial
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms.v2 as v2
# except:
#     import torchvision.transforms as v2

from tqdm import tqdm, trange
import pickle as pk
from collections import defaultdict
import traceback

from .data_utils import trivial_batch_collator, worker_init_reset_seed
from .tokenizer import make_tokenizer
# from ..helper import mv_functions
# from ..helper.utils import compute_t_for_f
from ..dist_utils import get_local_rank 
import time
import decord
decord.bridge.set_bridge('torch')



datasets = dict()
def register_dataset(name):
    def decorator(module):
        datasets[name] = module
        return module
    return decorator

# DOWNLOADED_FOLDER = {}
# def download_data(folder):
#     from pathlib import Path
#     from libs.helper import azure
#     from libs.dist_utils import barrier

#     target = folder[len('/tmp/zijia-2024/'):]
#     if os.path.exists(target):
#         print('Folder', target, 'already exists')
#     else:
#         target_parent = Path(target).parent
#         if get_local_rank() == 0:
#             print("Downloading", folder, "to", target)
#             os.makedirs(target_parent, exist_ok=True)
#             start = time.time()
#             cmd = azure.pull_command(str(target), str(target_parent))
#             azure.run_command(cmd)
#             print('download time', time.time()-start)
        
#     barrier()
#     return target

def parse_ego4d_files(ego4d_anno, ego4d_meta):

    with open(ego4d_meta, 'r') as f:
        metadata = json.load(f)
    v2meta = { v['video_uid']: v['video_metadata'] for v in metadata['videos'] }

    with open(ego4d_anno, 'r') as f:   
        anno = json.load(f)

    clip_info = {}
    for v in anno['videos']:
        for clip in v['clips']:
            d = {}
            for k in ['video_start_sec', 'video_end_sec']:
                d[k] = clip[k]
            d['video_uid'] = v['video_uid']

            # d['video_nframes'] = meta['num_frames']

            if v['video_uid'].startswith('grp'):
                h, w = 1440, 1920
            else:
                meta = v2meta[v['video_uid']]
                h, w = meta['display_resolution_height'], meta['display_resolution_width']
            d['height'] = h
            d['width'] = w

            clip_info[clip['clip_uid']] = d

    return clip_info

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label

def load_pk(fname, n=0):
    with open(fname, 'rb') as f:
        return pk.load(f)[n]

def load_pk_avg(fname):
    with open(fname, 'rb') as f:
        v_align, v_distill = pk.load(f)[:2]
    return ( v_align + v_distill ) /2

def full_pt(fname, dataset):
    vid = fname.split('/')[-1]
    path = fname.rsplit('/', 1)[0]

    info = dataset.video_info_for_mv[vid]
    feature = torch.load(path + '/' + info['video_uid'] + '.pt').numpy()
    div = lambda x: int( x * 30 // dataset.opt.data.clip_stride )
    start = max(0, div(info['video_start_sec']))
    end = min(div(info['video_end_sec']), feature.shape[0]+1)
    feature = feature[start:end]
    return feature

VID_LOAD_FUNC = {
    "npy" :  lambda x, d: np.load(x + '.npy').astype(np.float32),
    'pk1':   lambda x, d: load_pk(x + '.pk', 1),
    'pk0':   lambda x, d: load_pk(x + '.pk', 0),
    'pk_avg':   lambda x, d: load_pk_avg(x + '.pk'),
    'pt'  :  lambda x, d: torch.load(x + '.pt').numpy(),
    'full_pt': lambda x, d: full_pt(x, d),
}

def compute_action_label(annotation, mapping, fps, vid_dict, use_task=False):
    label2index, _ = load_action_mapping(mapping, sep="|")
    with open(annotation) as fp:
        full_train =  json.load(fp)

    # fps = 30 / 16
    # overlap = [0, 0]
    # bg = [0, 0]
    labeldict = {}
    raw_label = {}
    for video in full_train['videos']:
        # nframes = int(video['end_time'] * fps)
        nframes = vid_dict[video['video_uid']]['num_clips']
        label = [ label2index['background'] ] * nframes
        segs = sorted(video['segments'], key=lambda x: x['start_time']) 
        for seg in segs: 
            class_name = seg['step_category']
            if use_task:
                class_name = class_name.split(':')[0]

            # if seg['step_category'] not in label2index:
            #     print(f"Warning: {seg['step_category']} not in mapping")
            seg['class_label'] = label2index.get(class_name, label2index['background'])
            start = int(seg['start_time'] * fps)
            end = min(nframes, int(seg['end_time'] * fps))
            for t in range(start, end):
                label[t] = seg['class_label'] 
        labeldict[video['video_uid']] = torch.LongTensor(label)
        raw_label[video['video_uid']] = video
    return labeldict, raw_label

class BaseDataset(Dataset):

    def __init__(
        self,
        split,                  # data split, a tuple/list allowing concat of subsets
        is_training,            # whether in training mode
        
        anno_file,              # annotation json file
        vid_feat_dir,           # video feature directory
        text_feat_dir,          # text feature directory
        ext_score_dir,          # external score directory
        tokenizer,              # tokenizer (optional)

        clip_size,              # number of frames per clip / feature
        clip_stride,            # temporal stride of clips (in frame)
        downsample_rate=1,      # down-sampling rate for video features
        to_fixed_len=False,     # whether to resize video features to max length
        
        normalize_vid=False,    # whether to normalize video features to unit length
        normalize_text=False,   # whether to normalize text features to unit length
        normalize_scores=True,  # whether to normalize external score using sigmoid
        temperature=1.0,        # sigmoid temperature for score normalization

        max_vid_len=None,            # max video length (#clips) in training
        max_text_len=None,           # max text length (#tokens) in training

        crop_ratio=(0.9, 1.0),  # random cropping of video features in training
        trunc_thresh=0.5,       # threshold for event truncation in training
        max_num_text=None,      # max number of text queries per video in training
        
        group_method="greedy",  # text grouping method ("greedy" | "random" | "all")
        num_epochs=1,           # number of epochs
        **kwargs
    ):
        super(BaseDataset, self).__init__()

        self.opt = kwargs.get('opt', None)

        # assert os.path.exists(anno_file)
        if isinstance(split, str) and ',' in split:
            split = split.split(',')
        print(split)
        if not isinstance(split, (list, tuple)):
            split = (split, )
        if not isinstance(vid_feat_dir, (list, tuple)):
            vid_feat_dir = (vid_feat_dir, )
        # assert all([os.path.isdir(d) for d in vid_feat_dir])
        if tokenizer is None:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0  # this must be updated upon starting a new epoch

        # import ipdb; ipdb.set_trace()
        # self.opt.aux.mount = '/mnt/raptor/zijia/azure/decafnet/data' # HACK
        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir # [os.path.join(self.opt.aux.mount, v) for v in vid_feat_dir]
        self.text_feat_dir = text_feat_dir

        # self.text_feat_dir = os.path.join('/tmp/zijia-2024', text_feat_dir)
        self.ext_score_dir = ext_score_dir
        self.tokenizer = tokenizer

        self.max_vid_len = max_vid_len
        self.max_text_len = max_text_len
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate
        self.to_fixed_len = to_fixed_len

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.normalize_scores = normalize_scores
        self.temperature = temperature

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh
        self.max_num_text = max_num_text

        self.vid_dict, self.text_dict = self._parse_annotations()
        
        self.group_method = group_method
        self.num_epochs = num_epochs

        self.vid_feat_dict = {}
        self.text_feat_dict = {}

    def __mv_pretrain_init__(self, opt):
        self.input_size = opt.mv_enc.in_size        
        self.mv_stride = opt.mv_enc.in_stride
        # self.cw_size = opt.cmd.cw
        self.mv_transforms = v2.Compose([
                v2.Resize(size=self.input_size, antialias=True),
                v2.CenterCrop(size=(self.input_size, self.input_size)),
            ])

        # print('build samples time', time.time()-start)
        # splits = opt.data.eval_split if not self.is_training else opt.data.split

        # if not isinstance(splits, (list, tuple)):
        #     splits = (splits, )
        self.video_info_for_mv = {}
        for split in self.split:
            if split == 'train':
                info = parse_ego4d_files(opt.data.ego4d_train_anno, opt.data.ego4d_metadata)
                self.video_info_for_mv.update(info)
            if split == 'val':
                info = parse_ego4d_files(opt.data.ego4d_val_anno, opt.data.ego4d_metadata)
                self.video_info_for_mv.update(info)

        # self.mv_dir = opt.data.mv_dir
# 


    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        # combine data from all splits
        anno_db = dict()
        for s in self.split:
            assert s in anno, 'split [{:s}] does not exist'.format(s)
            anno_db.update(anno[s])

        dup_ct = 0
        vid_dict, text_dict = OrderedDict(), OrderedDict()
        for key, value in anno_db.items():
            if 'annotations' not in value:
                continue

            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps
            
            if 'num_clips' in value:
                # ds = self.opt.data.true_ds if self.opt.data.true_ds is not None else self.opt.data.downsample_rate
                ds = self.opt.data.downsample_rate
                num_clips = ( value['num_clips'] + ds - 1 ) // ds
            else:
                num_clips = None

            text_ids, segments = tuple(), tuple()
            for s, pair in enumerate(value['annotations']):
                start = max(float(pair['segment'][0]), 0)
                end = min(float(pair['segment'][1]), duration)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                segment = (start, end)

                text = pair['sentence'].strip()
                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))
                text_ids += (text_id, )
                segments += (segment, )

                text_dict[text_id] = {
                    'text'      : text,
                    'segment'   : np.array(segment)[None],
                    'text_idx'  : s,
                    'vid_id'    : key,
                }
            
            if len(text_ids) == 0:
                continue
        
            if len(set(text_ids)) < len(text_ids):
                # print('duplicate text ids in video {:s}'.format(key))
                dup_ct += 1
                # import ipdb; ipdb.set_trace()

            vid_dict[key] = {
                'fps'       : fps,
                'num_frames': num_frames,
                'num_clips' : num_clips,
                'duration'  : duration,
                'text_ids'  : text_ids,
                'segments'  : np.array(segments),
                'annotations': value['annotations'],
            }

        print('duplicate text ids in {:d} videos'.format(dup_ct))

        # HACK
        text_dict = None

        return vid_dict, text_dict

    def _load_vid_feats(self, vid_id):
        if vid_id in self.vid_feat_dict:
            return self.vid_feat_dict[vid_id]

        # vid_feat_files = [os.path.join(d, vid_id + '.npy') \
        #     for d in self.vid_feat_dir]
        # vid_feats = [np.load(f).astype(np.float32) for f in vid_feat_files]
        vid_feat_files = [os.path.join(d, vid_id) for d in self.vid_feat_dir]
        vid_feats = [ VID_LOAD_FUNC[self.opt.data.vid_load](f, self) 
                        for f in vid_feat_files]

        # assume features from different sources are apporoximately aligned
        # (flow features may be one unit shorter than RGB features)
        if len(vid_feats) > 1:
            feat_lens = [len(x) for x in vid_feats]
            max_len, min_len = max(feat_lens), min(feat_lens)
            assert max_len - min_len <= 10, \
                'misaligned features ([max] {:d}, [min] {:d}) for video {:s}' \
                ''.format(max_len, min_len, vid_id)

            # pad shorter sequences by replicating last feature vector
            for idx in range(len(vid_feats)):
                if feat_lens[idx] < max_len:
                    pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                    vid_feats[idx] = np.concatenate((vid_feats[idx], pad))

            # concatenate features along channel dimension
            vid_feats = np.concatenate(vid_feats, axis=-1)  # (t, c)
        else:
            vid_feats = vid_feats[0]

        # temporally down-sample features
        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)

        self.vid_feat_dict[vid_id] = vid_feats

        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,          # float tensor (c, t), full video features 
        segments,       # float tensor (n, 2), event segments
        offset,         # float, clip offset
        num_trials=5000 # int, number of trials
    ):
        vid_len = feats.size(1)
        max_vid_len = self.max_vid_len

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, segments, None

            max_vid_len = random.randint(
                max(np.ceil(self.crop_ratio[0] * vid_len), 1),
                min(np.ceil(self.crop_ratio[1] * vid_len), vid_len)
            )
            if max_vid_len == vid_len:
                return feats, segments, None

        # rough estimate on the range of valid chunks
        s0 = max(0, np.floor(segments[:, 0].max() - max_vid_len))
        s1 = min(vid_len - max_vid_len, np.ceil(segments[:, 1].min()))
        
        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        if seg_lens.max() > (self.max_vid_len / self.trunc_thresh): 
            trunc_thresh = 0.2
            print(f'data is too long {seg_lens.max()}, change truncate threshold to {trunc_thresh}')
        else:
            trunc_thresh = self.trunc_thresh

        for _ in range(num_trials):
            ws = random.randint(s0, s1) # window start
            we = ws + max_vid_len       # window end

            # check overlap with segments
            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > trunc_thresh):
                feats = feats[:, ws:we]
                segments = torch.clamp(
                    segments - ws, min=-offset, max=we - ws + offset
                )
                return feats, segments, [ws, we]
            
        # import ipdb; ipdb.set_trace()
        raise ValueError('no valid truncation found')

    def _load_text_feats(self, text_id):
        if text_id in self.text_feat_dict:
            return self.text_feat_dict[text_id]

        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        else:
            try:
                text_feat_file = os.path.join(self.text_feat_dir, str(text_id) + '.npy')
                text_feats = np.load(text_feat_file).astype(np.float32)
            except:
                raise ValueError(
                    f'failed to load features for sentence {text_id}'
                )
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def _load_ext_scores(self, text_id):
        try:
            score_file = os.path.join(self.ext_score_dir, text_id + '.npy')
            scores = np.load(score_file).astype(np.float32)
        except:
            raise ValueError(
                'failed to load external scores for sentence {:s}'.format(text_id)
            )

        # temporally down-sample scores
        if self.downsample_rate > 1:
            scores = scores[::self.downsample_rate]

        scores = torch.from_numpy(np.ascontiguousarray(scores))[None]   # (1, t)

        if self.normalize_scores:
            scores = torch.sigmoid(scores / self.temperature)

        self.text_feat_dict[text_id] = scores

        return scores

    def _avgpool_to_fixed_len(self, feats, size):
        vid_len = feats.size(1)
        sampling_ratio = math.ceil(vid_len / size)
        feats = F.interpolate(
            feats[None],
            size=size * sampling_ratio, mode='linear', align_corners=False
        )
        if sampling_ratio > 1:
            feats = F.avg_pool1d(feats, kernel_size=sampling_ratio)
        feats = feats[0]

        return feats

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


@register_dataset('video_centric')
class VideoCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video and a subset of its associated text queries.

    NOTE: this class behaves correctly in training only when all processes
    share exactly the same random seed during initialization. This allows
    identical grouping of training samples across all processes.

    Expected behavior:
    - train: a video + no more than max_num_text text queries
    - eval: a video + all of its text queries
    """
    MV_DOWNLOADED = False

    def __init__(self, **kwargs):
        super(VideoCentricDataset, self).__init__(**kwargs)

        if self.is_training:
            self.data_list = self._build_train_samples()
        else:
            assert self.num_epochs == 1
            self.data_list = self._build_eval_samples()

        self.text_cls_dict = {}
        for split in self.split:
            fname = self.opt.data.text_cls_fname.format(split=split)
            cls_dict = np.load(fname, allow_pickle=True).item()
            self.text_cls_dict.update(cls_dict)

        # self.download_mv_feat()
    
    # def download_mv_feat(self):
    #     from pathlib import Path
    #     from libs.helper import azure
    #     from libs.dist_utils import barrier

    #     # print(self.opt.data.mv_feat_dir)
    #     if self.opt.data.mv_feat_dir is not None and self.opt.aux.download_mv_feat:
    #         target = self.opt.data.mv_feat_dir[len('/tmp/zijia-2024/'):]
    #         self.opt.data.mv_feat_dir = str(target)
    #         if os.path.exists(target):
    #             print('Folder', target, 'already exists')
    #         else:
    #             target_parent = Path(target).parent
    #             if get_local_rank() == 0 and (not VideoCentricDataset.MV_DOWNLOADED):
    #                 print("Downloading mv feature from", self.opt.data.mv_feat_dir, "to", target)
    #                 os.makedirs(target_parent, exist_ok=True)
    #                 start = time.time()
    #                 cmd = azure.pull_command(str(target), str(target_parent))
    #                 azure.run_command(cmd)
    #                 print('download time', time.time()-start)
    #                 VideoCentricDataset.MV_DOWNLOADED = True
            
    #     barrier()


    def _build_train_samples(self):
        samples = []
        ## NOTE: here we pre-calculated samples for all epochs
        ## we do not shuffle samples here and rely on sampler to do it
        for _ in trange(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                samples.extend(self._group(vid_id))
        samples = samples[:len(samples) // self.num_epochs * self.num_epochs]
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples.extend( [(vid_id, tuple(range(len(vid_dict['segments']))))] )
        return tuple(samples)

    def _group(self, vid_id):
        if self.to_fixed_len:
            return self._group_with_fixed_len(vid_id)
        return self._group_with_max_len(vid_id)

    def _group_with_fixed_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]
        idx = list(range(len(vid_dict['segments'])))

        if self.group_method in ("random", "all"):
            return [(vid_id, tuple(idx))]

        random.shuffle(idx)
        samples = []
        for i in range(0, len(idx), self.max_num_text):
            sample = (vid_id, tuple(idx[i:i + self.max_num_text]))
            samples += [sample]
        return samples

    def _group_with_max_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]

        # worse-case window size
        if vid_dict['num_clips'] <= self.max_vid_len: # = ~2304
            win_len = vid_dict['num_clips']
            if self.crop_ratio is not None:
                win_len = max(np.ceil(self.crop_ratio[0] * win_len), 1)
        else:
            win_len = self.max_vid_len
        win_len = (
            self.clip_stride * (win_len - 1) + self.clip_size
        ) / vid_dict['fps'] # window length in seconds

        # sort segments in ascending order of start time
        sort_idx = np.argsort(vid_dict['segments'][:, 0])
        segments = vid_dict['segments'][sort_idx]
        mask = np.ones(len(segments), dtype=bool)

        samples = []
        while mask.sum() > 0:
            # probe selection
            ## NOTE: our heuristic is to always select 1st available segment
            ptr = np.nonzero(mask)[0].min()

            # largest window covering probe
            ## NOTE: here we do not consider truncation effect for simplicity
            ## this also adds some room for temporal jittering in data loading
            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])   # corner case: segment longer than window
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )   # candidates fully covered by window
                idx = np.nonzero(is_inside)[0]
                if len(idx) > self.max_num_text:
                    # sample a subset if too many candidates
                    idx = np.random.choice(idx, self.max_num_text, replace=False)
            sample = (vid_id, tuple(sort_idx[idx]))
            samples += [sample]
            mask[idx] = 0
        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def _load_text_cls_feats(self, vid_id, seg_idx):
        info = self.vid_dict[vid_id]
        query = info['annotations']
        query = [ query[i] for i in seg_idx ]
        query = [ self.text_cls_dict[q['sentence']] for q in query ]
        query = np.concatenate(query, axis=0)
        query = torch.from_numpy(query) #.float()
        return query

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        for i in range(10):
            try:
                vid_feats = self._load_vid_feats(vid_id)
                break
            except Exception as e:
                if i == 9:
                    raise e
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride

        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            vid_dict['segments'][np.array(seg_idx)] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        se = None
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments, se = self._truncate_vid_feats(
                    vid_feats, segments, clip_offset
                )
            if self.group_method == "random" and len(seg_idx) > self.max_num_text:
                seg_idx = random.sample(seg_idx, k=self.max_num_text)
                segments = segments[seg_idx]

        # load text features / IDs
        text_feats_list = tuple()
        for idx in seg_idx:
            for i in range(10):
                try:
                    text_feats = self._load_text_feats(vid_dict['text_ids'][idx])
                    break
                except Exception as e:
                    if i == 9:
                        raise e
            text_feats_list += (text_feats, )

        text_cls_feats = self._load_text_cls_feats(vid_id, seg_idx)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores_list = tuple()
            for idx in seg_idx:
                scores = self._load_ext_scores(vid_dict['text_ids'][idx])
                if self.to_fixed_len:
                    scores = self._avgpool_to_fixed_len(
                        scores, self.max_vid_len
                    )
                ext_scores_list += (scores, )
            ext_scores = torch.cat(ext_scores_list)
        else:
            ext_scores = None

        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : vid_dict['segments'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segments in grid unit
                 'clip_id'    : vid_id,
                 'text_id'    : seg_idx,

                 'vid'        : vid_feats,              # video features (c2, t2)
                 'text'       : text_feats_list,        # text features List[(c1, t1) x n]
                 'text_cls'   : text_cls_feats,         # text cls features (n, c1)
                 'ext_scores' : ext_scores,             # external scores (n, t2)
                #  'mv_data'    : mv_feats,
                }


@register_dataset('video_centric_clip')
class VideoCentricCLIPDataset(VideoCentricDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.text_id2text = {}
        for vname, vinfo in self.vid_dict.items():
            for i, q in enumerate(vinfo['annotations']):
                self.text_id2text[q['sentence_id']] = q['sentence']
        # self.raw_text_feat_dict = np.load('/tmp/zijia-2024/datasets/clip/nlq_token_features_B_32.npy', allow_pickle=True).item()
        self.raw_text_feat_dict = np.load(self.opt.data.clip_token_fname, allow_pickle=True).item()

        self.tokenizer = None

    def _load_text_feats(self, text_id):
        if text_id in self.text_feat_dict:
            return self.text_feat_dict[text_id]

        text_feats = self.raw_text_feat_dict[self.text_id2text[text_id]]
        text_feats = text_feats.transpose()     # (c, t)
        text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

@register_dataset('video_centric_clip_twofeat')
class VideoCentricCLIPTwoFeatDataset(VideoCentricDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.text_id2text = {}
        for vname, vinfo in self.vid_dict.items():
            for i, q in enumerate(vinfo['annotations']):
                self.text_id2text[q['sentence_id']] = q['sentence']
        # self.raw_text_feat_dict = np.load('/tmp/zijia-2024/datasets/clip/nlq_token_features_B_32.npy', allow_pickle=True).item()
        self.raw_text_feat_dict = np.load(self.opt.data.clip_token_fname, allow_pickle=True).item()

        self.tokenizer = None
        self.shallow_vid_feat_dict = {}

        # self.shallow_vid_feat_dir = [os.path.join(self.opt.aux.mount, v) for v in self.opt.data.shallow_vid_feat_dir]
        self.shallow_vid_feat_dir = self.opt.data.shallow_vid_feat_dir

    def _load_text_feats(self, text_id):
        if text_id in self.text_feat_dict:
            return self.text_feat_dict[text_id]

        text_feats = self.raw_text_feat_dict[self.text_id2text[text_id]]
        text_feats = text_feats.transpose()     # (c, t)
        text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def _load_shallow_vid_feats(self, vid_id):
        # assert self.downsample_rate == 2

        if vid_id in self.shallow_vid_feat_dict:
            return self.shallow_vid_feat_dict[vid_id]

        try:
            # vid_feat_files = [os.path.join(d, vid_id + '.npy') \
            #     for d in self.vid_feat_dir]
            # vid_feats = [np.load(f).astype(np.float32) for f in vid_feat_files]
            vid_feat_files = [os.path.join(d, vid_id) for d in self.shallow_vid_feat_dir]
            vid_feats = [ VID_LOAD_FUNC[self.opt.data.shallow_vid_load](f, self) 
                            for f in vid_feat_files]
        except:
            raise ValueError(
                'failed to load features for video {:s}'.format(vid_id)
            )

        # assume features from different sources are apporoximately aligned
        # (flow features may be one unit shorter than RGB features)
        if len(vid_feats) > 1:
            feat_lens = [len(x) for x in vid_feats]
            max_len, min_len = max(feat_lens), min(feat_lens)
            assert max_len - min_len <= 10, \
                'misaligned features ([max] {:d}, [min] {:d}) for video {:s}' \
                ''.format(max_len, min_len, vid_id)

            # pad shorter sequences by replicating last feature vector
            for idx in range(len(vid_feats)):
                if feat_lens[idx] < max_len:
                    pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                    vid_feats[idx] = np.concatenate((vid_feats[idx], pad))

            # concatenate features along channel dimension
            vid_feats = np.concatenate(vid_feats, axis=-1)  # (t, c)
        else:
            vid_feats = vid_feats[0]

        # temporally down-sample features
        if self.opt.data.shallow_ds > 1:
            vid_feats = vid_feats[::self.opt.data.shallow_ds]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)

        self.shallow_vid_feat_dict[vid_id] = vid_feats

        return vid_feats

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        for i in range(10):
            try:
                vid_feats = self._load_vid_feats(vid_id)
                break
            except Exception as e:
                raise e

        for i in range(10):
            try:
                shallow_vid_feats = self._load_shallow_vid_feats(vid_id)
                break
            except Exception as e:
                raise e

        vid_len = min(shallow_vid_feats.size(1), vid_feats.size(1))

        shallow_vid_feats = shallow_vid_feats[:, :vid_len]
        vid_feats = vid_feats[:, :vid_len]

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride

        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            vid_dict['segments'][np.array(seg_idx)] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        se = None
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments, se = self._truncate_vid_feats(
                    vid_feats, segments, clip_offset
                )
                if se is not None:
                    shallow_vid_feats = shallow_vid_feats[:, se[0]:se[1]]
            if self.group_method == "random" and len(seg_idx) > self.max_num_text:
                seg_idx = random.sample(seg_idx, k=self.max_num_text)
                segments = segments[seg_idx]

        # load text features / IDs
        text_feats_list = tuple()
        for idx in seg_idx:
            for i in range(10):
                try:
                    text_feats = self._load_text_feats(vid_dict['text_ids'][idx])
                    break
                except Exception as e:
                    raise e
            text_feats_list += (text_feats, )

        text_cls_feats = self._load_text_cls_feats(vid_id, seg_idx)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores_list = tuple()
            for idx in seg_idx:
                scores = self._load_ext_scores(vid_dict['text_ids'][idx])
                if self.to_fixed_len:
                    scores = self._avgpool_to_fixed_len(
                        scores, self.max_vid_len
                    )
                ext_scores_list += (scores, )
            ext_scores = torch.cat(ext_scores_list)
        else:
            ext_scores = None

        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : vid_dict['segments'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segments in grid unit
                 'clip_id'    : vid_id,
                 'text_id'    : seg_idx,

                 'vid'         : vid_feats,              # video features (c2, t2)
                 'shallow_vid' : shallow_vid_feats,              # video features (c2, t2)
                 'text'       : text_feats_list,        # text features List[(c1, t1) x n]
                 'text_cls'   : text_cls_feats,         # text cls features (n, c1)
                 'ext_scores' : ext_scores,             # external scores (n, t2)
                #  'mv_data'    : mv_feats,
                }

@register_dataset('video_centric_twofeat')
class VideoCentricTwoFeatDataset(VideoCentricDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.text_id2text = {}
        # for vname, vinfo in self.vid_dict.items():
        #     for i, q in enumerate(vinfo['annotations']):
        #         self.text_id2text[q['sentence_id']] = q['sentence']
        # self.raw_text_feat_dict = np.load('/tmp/zijia-2024/datasets/clip/nlq_token_features_B_32.npy', allow_pickle=True).item()

        self.tokenizer = None
        self.shallow_vid_feat_dict = {}

        # self.shallow_vid_feat_dir = [os.path.join(self.opt.aux.mount, v) for v in self.opt.data.shallow_vid_feat_dir]
        self.shallow_vid_feat_dir = self.opt.data.shallow_vid_feat_dir

    # def _load_text_feats(self, text_id):
    #     if text_id in self.text_feat_dict:
    #         return self.text_feat_dict[text_id]

    #     text_feats = self.raw_text_feat_dict[self.text_id2text[text_id]]
    #     text_feats = text_feats.transpose()     # (c, t)
    #     text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
    #     if self.is_training:
    #         text_feats = text_feats[:, :self.max_text_len]

    #     # normalize text features to unit length
    #     if self.normalize_text:
    #         text_feats = F.normalize(text_feats, dim=0)

    #     return text_feats

    def _load_shallow_vid_feats(self, vid_id):
        assert self.downsample_rate == 2

        if vid_id in self.shallow_vid_feat_dict:
            return self.shallow_vid_feat_dict[vid_id]

        try:
            # vid_feat_files = [os.path.join(d, vid_id + '.npy') \
            #     for d in self.vid_feat_dir]
            # vid_feats = [np.load(f).astype(np.float32) for f in vid_feat_files]
            vid_feat_files = [os.path.join(d, vid_id) for d in self.shallow_vid_feat_dir]
            vid_feats = [ VID_LOAD_FUNC[self.opt.data.shallow_vid_load](f, self) 
                            for f in vid_feat_files]
        except:
            raise ValueError(
                'failed to load features for video {:s}'.format(vid_id)
            )

        # assume features from different sources are apporoximately aligned
        # (flow features may be one unit shorter than RGB features)
        if len(vid_feats) > 1:
            feat_lens = [len(x) for x in vid_feats]
            max_len, min_len = max(feat_lens), min(feat_lens)
            assert max_len - min_len <= 10, \
                'misaligned features ([max] {:d}, [min] {:d}) for video {:s}' \
                ''.format(max_len, min_len, vid_id)

            # pad shorter sequences by replicating last feature vector
            for idx in range(len(vid_feats)):
                if feat_lens[idx] < max_len:
                    pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                    vid_feats[idx] = np.concatenate((vid_feats[idx], pad))

            # concatenate features along channel dimension
            vid_feats = np.concatenate(vid_feats, axis=-1)  # (t, c)
        else:
            vid_feats = vid_feats[0]

        # temporally down-sample features
        # if self.downsample_rate > 1:
        #     vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)

        self.shallow_vid_feat_dict[vid_id] = vid_feats

        return vid_feats

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        for i in range(10):
            try:
                vid_feats = self._load_vid_feats(vid_id)
                break
            except Exception as e:
                raise e

        for i in range(10):
            try:
                shallow_vid_feats = self._load_shallow_vid_feats(vid_id)
                break
            except Exception as e:
                raise e

        vid_len = min(shallow_vid_feats.size(1), vid_feats.size(1))

        shallow_vid_feats = shallow_vid_feats[:, :vid_len]
        vid_feats = vid_feats[:, :vid_len]

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride

        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            vid_dict['segments'][np.array(seg_idx)] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        se = None
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments, se = self._truncate_vid_feats(
                    vid_feats, segments, clip_offset
                )
            if self.group_method == "random" and len(seg_idx) > self.max_num_text:
                seg_idx = random.sample(seg_idx, k=self.max_num_text)
                segments = segments[seg_idx]

        # load text features / IDs
        text_feats_list = tuple()
        for idx in seg_idx:
            for i in range(10):
                try:
                    text_feats = self._load_text_feats(vid_dict['text_ids'][idx])
                    break
                except Exception as e:
                    raise e
            text_feats_list += (text_feats, )

        text_cls_feats = self._load_text_cls_feats(vid_id, seg_idx)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores_list = tuple()
            for idx in seg_idx:
                scores = self._load_ext_scores(vid_dict['text_ids'][idx])
                if self.to_fixed_len:
                    scores = self._avgpool_to_fixed_len(
                        scores, self.max_vid_len
                    )
                ext_scores_list += (scores, )
            ext_scores = torch.cat(ext_scores_list)
        else:
            ext_scores = None

        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : vid_dict['segments'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segments in grid unit
                 'clip_id'    : vid_id,
                 'text_id'    : seg_idx,

                 'vid'         : vid_feats,              # video features (c2, t2)
                 'shallow_vid' : shallow_vid_feats,              # video features (c2, t2)
                 'text'       : text_feats_list,        # text features List[(c1, t1) x n]
                 'text_cls'   : text_cls_feats,         # text cls features (n, c1)
                 'ext_scores' : ext_scores,             # external scores (n, t2)
                #  'mv_data'    : mv_feats,
                }


class DataBank():

    def __init__(self):
        self.data = OrderedDict()
    
    def get(self, key):
        return self.data.get(key, None)
    
    def set(self, key, value):
        # print('Data bank set', key)
        self.data[key] = value

    def purge(self, n=0):
        del self.data
        self.data = OrderedDict()
        # if len(self.data) > n:
        #     # print('purge')
        #     self.data = OrderedDict(list(self.data.items())[-n:])




@register_dataset('action_recog_joint')
class JointDataset(BaseDataset):

    def __init__(self, opt, tokenizer, is_training, num_epochs, **data_opts):
        self.opt = opt
        self.is_training = is_training  

        print('Training: ', is_training, 'Dataset Goalstep')
        self.goalstep = ActionRecognitionDataset(opt=opt, tokenizer=tokenizer, is_training=is_training, num_epochs=num_epochs, action_label=True, **opt.data)

        nlq_opt = opt.clone()
        nlq_opt.data.ego4d_train_anno = '/tmp/zijia-2024/datasets/ego4d_data/v2/annotations/nlq_train.json'
        nlq_opt.data.ego4d_val_anno   = '/tmp/zijia-2024/datasets/ego4d_data/v2/annotations/nlq_val.json'
        nlq_opt.data.ego4d_metadata   = '/tmp/zijia-2024/datasets/ego4d_data/ego4d_v2.json'
        nlq_opt.data.mv_dir           = '/tmp/zijia-2024/datasets/ego4d_nlq_mv_partition'
        nlq_opt.data.mv_img_dir       = '/tmp/zijia-2024/datasets/ego4d_nlq_mv_img_combined_small'
        nlq_opt.data.video_dir        = '/tmp/zijia-2024/datasets/ego4d_video_256/'
        nlq_opt.data.anno_file        = '/tmp/zijia-2024/datasets/egovlpv1/features/v1/nlq_32x8_d2.json'
        nlq_opt.data.vid_feat_dir     = '/tmp/zijia-2024/datasets/egovlpv1/features/v1/video/32x8_d2'
        nlq_opt.data.text_feat_dir    = '/tmp/zijia-2024/datasets/egovlpv1/features/v1/text/text_768d/'

        print('Training: ', is_training, 'Dataset NLQ')
        self.nlq = ActionRecognitionDataset(opt=nlq_opt, tokenizer=tokenizer, is_training=is_training, num_epochs=num_epochs, action_label=False, **nlq_opt.data)

        self.epoch = 0  # this must be updated upon starting a new epoch

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.nlq.set_epoch(epoch)
        self.goalstep.set_epoch(epoch)

    def __len__(self):
        return len(self.nlq) + len(self.goalstep)

    def __getitem__(self, idx):
        # idx = self.epoch * len(self) + idx
        if idx < len(self.nlq):
            return self.nlq[idx]
        else:
            return self.goalstep[idx - len(self.nlq)]


def make_dataset(opt, num_epochs=1, is_training=True):
    opt = deepcopy(opt)
    if not is_training:
        opt.data = opt.eval.data
    data_opt = opt.data
    # data_opt = opt['data'] 
    # if not is_training:
    #     data_opt['split'] = data_opt['eval_split']
        
    if 'tokenizer' in data_opt:
        tokenizer = make_tokenizer(data_opt.pop('tokenizer'))
    else:
        tokenizer = None
    
    if is_training:
        dataset_name = data_opt.pop('name') 
    elif data_opt.get('eval_name', None) is None:
        dataset_name = data_opt.pop('name')
    else:
        dataset_name = data_opt.get('eval_name')
    print('Dataset Name: ', dataset_name)

    return datasets[dataset_name](
        opt = opt,
        tokenizer=tokenizer, is_training=is_training, num_epochs=num_epochs,  **data_opt
    )


def make_dataloader(
    dataset,            # dataset
    generator,          # random number generator that controls worker seed
    batch_size,         # local batch size
    num_workers,        # local number of workers
    is_training,        # whether is in training
    world_size=1,       # number of processes (GPUs)
    rank=0,             # current process
    # prefetch_factor=2,  # number of batches prefetch
):
    sampler = None
    # if world_size > 1 and is_training:
    #     sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=is_training)
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=is_training)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, rank),
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
        # prefetch_factor=prefetch_factor,
    )
    return loader, sampler


