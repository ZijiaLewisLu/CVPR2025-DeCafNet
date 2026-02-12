from collections import OrderedDict
import json
import os

import numpy as np
import torch

from ..utils.dataset_utils import make_tokenizer
from ..utils.dataset_utils import (
    create_dataset_registry,
    VID_LOAD_FUNC,
    make_dataset_from_registry,
    make_dataloader_common,
)
from ..utils.dataset_utils import BaseGroundingDataset


datasets, register_dataset = create_dataset_registry()


@register_dataset('mad')
class MADDataset(BaseGroundingDataset):
    def __init__(
        self,
        split,
        is_training,
        anno_file,
        vid_feat_dir,
        text_feat_dir,
        ext_score_dir,
        tokenizer,
        clip_size,
        clip_stride,
        downsample_rate=1,
        to_fixed_len=False,
        normalize_vid=False,
        normalize_text=False,
        normalize_scores=True,
        temperature=1.0,
        max_vid_len=None,
        max_text_len=None,
        crop_ratio=(0.9, 1.0),
        trunc_thresh=0.5,
        max_num_text=None,
        group_method='greedy',
        num_epochs=1,
        **kwargs,
    ):
        super().__init__()
        self.opt = kwargs.get('opt', None)

        self._setup_common(
            split=split,
            is_training=is_training,
            anno_file=anno_file,
            vid_feat_dir=vid_feat_dir,
            text_feat_dir=text_feat_dir,
            ext_score_dir=ext_score_dir,
            tokenizer=tokenizer,
            clip_size=clip_size,
            clip_stride=clip_stride,
            downsample_rate=downsample_rate,
            to_fixed_len=to_fixed_len,
            normalize_vid=normalize_vid,
            normalize_text=normalize_text,
            normalize_scores=normalize_scores,
            temperature=temperature,
            max_vid_len=max_vid_len or 2304,
            max_text_len=max_text_len or 256,
            crop_ratio=crop_ratio,
            trunc_thresh=trunc_thresh or 0.5,
            max_num_text=max_num_text or 10,
            group_method=group_method,
            num_epochs=num_epochs,
        )

        self.vid_dict = self._parse_annotations()
        self._build_data_list(assert_eval_single_epoch=True)

        if (
            self.opt
            and hasattr(self.opt, 'data')
            and hasattr(self.opt.data, 'text_cls_fname')
            and self.opt.data.text_cls_fname is not None
        ):
            try:
                text_cls_data = np.load(self.opt.data.text_cls_fname, allow_pickle=True)
                if (
                    isinstance(text_cls_data, np.ndarray)
                    and text_cls_data.dtype == object
                    and text_cls_data.shape == ()
                ):
                    self.text_cls_dict = dict(text_cls_data.item())
                elif hasattr(text_cls_data, 'files'):
                    self.text_cls_dict = {k: text_cls_data[k] for k in text_cls_data.files}
                else:
                    self.text_cls_dict = None
            except Exception:
                self.text_cls_dict = None
        else:
            self.text_cls_dict = None

    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        anno_db = dict()
        for s in self.split:
            assert s in anno, 'split [{:s}] does not exist'.format(s)
            anno_db.update(anno[s])

        dup_ct = 0
        vid_dict = OrderedDict()
        for key, value in anno_db.items():
            if 'annotations' not in value:
                continue

            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps

            if 'num_clips' in value:
                ds = self.opt.data.downsample_rate
                num_clips = (value['num_clips'] + ds - 1) // ds
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

                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))
                text_ids += (text_id,)
                segments += (segment,)

            if len(text_ids) == 0:
                continue
            if len(set(text_ids)) < len(text_ids):
                dup_ct += 1

            vid_dict[key] = {
                'fps': fps,
                'num_frames': num_frames,
                'num_clips': num_clips,
                'duration': duration,
                'text_ids': text_ids,
                'segments': np.array(segments),
                'annotations': value['annotations'],
            }

        print('duplicate text ids in {:d} videos'.format(dup_ct))
        return vid_dict

    def _load_vid_feats(self, vid_id):
        if vid_id in self.vid_feat_dict:
            return self.vid_feat_dict[vid_id]

        vid_feat_files = [os.path.join(d, vid_id) for d in self.vid_feat_dir]
        vid_feats = [VID_LOAD_FUNC[self.opt.data.vid_load](f, self) for f in vid_feat_files]
        vid_feats = self._merge_video_feat_sources(vid_feats, vid_id, assert_aligned=True)
        vid_feats = self._to_video_tensor(vid_feats, self.downsample_rate)

        self.vid_feat_dict[vid_id] = vid_feats
        return vid_feats

    def _load_text_cls_feats(self, vid_id, seg_idx):
        info = self.vid_dict[vid_id]
        query = info['annotations']
        query = [query[i] for i in seg_idx]
        query = [self.text_cls_dict[q['sentence']] for q in query]
        query = np.stack(query, axis=0)
        query = torch.from_numpy(query)
        return query

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        vid_feats = self._with_retry(self._load_vid_feats, vid_id)
        vid_feats, _, clip_size, clip_stride, clip_offset = self._prepare_video_item(vid_feats)
        segments = self._compute_target_segments(vid_dict, seg_idx, clip_stride, clip_offset)
        vid_feats, segments, seg_idx, _ = self._maybe_truncate_and_sample(
            vid_feats,
            segments,
            clip_offset,
            seg_idx,
        )

        text_feats_list = self._load_text_feature_list(vid_dict, seg_idx)
        text_cls_feats = self._load_text_cls_feats(vid_id, seg_idx)
        ext_scores = self._load_ext_scores_list(vid_dict, seg_idx)

        return {
            'fps': vid_dict['fps'],
            'num_frames': vid_dict['num_frames'],
            'duration': vid_dict['duration'],
            'segment': vid_dict['segments'],
            'clip_size': clip_size,
            'clip_stride': clip_stride,
            'target': segments,
            'clip_id': vid_id,
            'text_id': seg_idx,
            'vid': vid_feats,
            'text': text_feats_list,
            'text_cls': text_cls_feats,
            'ext_scores': ext_scores,
        }


def make_dataset(opt, num_epochs=1, is_training=True):
    return make_dataset_from_registry(
        opt=opt,
        datasets=datasets,
        make_tokenizer_fn=make_tokenizer,
        num_epochs=num_epochs,
        is_training=is_training,
    )


def make_dataloader(
    dataset,
    generator,
    batch_size,
    num_workers,
    is_training,
    world_size=1,
    rank=0,
):
    return make_dataloader_common(
        dataset,
        generator=generator,
        batch_size=batch_size,
        num_workers=num_workers,
        is_training=is_training,
        world_size=world_size,
        rank=rank,
    )
