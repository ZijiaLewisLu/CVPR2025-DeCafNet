from collections import OrderedDict, defaultdict
import json
import os

import decord
import numpy as np
import torch
import torchvision.transforms.v2 as v2

from ..utils.dataset_utils import (
    BaseGroundingDataset,
    VID_LOAD_FUNC as COMMON_VID_LOAD_FUNC,
    create_dataset_registry,
    make_dataloader_common,
    make_dataset_from_registry,
    make_tokenizer,
)


decord.bridge.set_bridge("torch")


datasets, register_dataset = create_dataset_registry()


EGO4D_BAD_VIDEO_ID = "034b7e9d-7ce5-4b5c-8fab-18c08981952b"
EGO4D_BAD_SENTENCE = "who did I talked to when I operated phone?"
DEFAULT_TEXT_KEYS = ("sentence", "query", "text", "caption", "description")


@register_dataset("pretraining")
class PretrainingDataset(BaseGroundingDataset):
    def __init__(
        self,
        split,
        is_training,
        anno_file=None,
        vid_feat_dir=None,
        text_feat_dir=None,
        ext_score_dir=None,
        tokenizer=None,
        clip_size=32,
        clip_stride=8,
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
        group_method="greedy",
        num_epochs=1,
        **kwargs,
    ):
        super().__init__()
        self.opt = kwargs.get("opt", None)
        self.annotation_text_keys = tuple(self.opt.data.get("annotation_text_keys", DEFAULT_TEXT_KEYS))
        self.video_fname_template = self.opt.data.get("video_fname_template", "{video_uid}.mp4")
        self.decode_fps = float(self.opt.data.get("decode_fps", 30.0))
        self.full_pt_fps = float(self.opt.data.get("full_pt_fps", self.decode_fps))

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
            max_vid_len=max_vid_len,
            max_text_len=max_text_len,
            crop_ratio=crop_ratio,
            trunc_thresh=trunc_thresh,
            max_num_text=max_num_text,
            group_method=group_method,
            num_epochs=num_epochs,
        )

        self.anno_file = anno_file
        if self.anno_file is None:
            raise ValueError("missing annotation file. set data.anno_file")
        self.vid_dict, self.text_dict = self._parse_annotations(self.anno_file)
        self.video_info = self._build_video_info()

        f = np.arange(100000)
        clip_offset = int(0.5 * self.clip_size / self.clip_stride)
        f2i = (f + clip_offset) * self.clip_stride
        self.f2i_start = f2i - self.clip_size // 2
        self.f2i_end = f2i + self.clip_size // 2

        self.transforms = v2.Compose(
            [
                v2.Resize(self.opt.data.img_size),
                v2.CenterCrop(self.opt.data.img_size),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.text_cls_dict = self._load_text_cls_features()

        self._create_label()
        self._build_data_list(assert_eval_single_epoch=False)

    def _resolve_split_file(self, path_or_map, split):
        if path_or_map is None:
            return None
        if isinstance(path_or_map, dict):
            path = path_or_map.get(split, path_or_map.get("default"))
        else:
            path = path_or_map
        if path is None:
            return None
        if "{split}" in path:
            return path.format(split=split)
        return path

    def _load_text_cls_features(self):
        text_cls_dict = {}
        loaded_paths = set()
        text_cls_path_cfg = self.opt.data.get("text_cls_fname")
        for split in self.split:
            text_cls_file = self._resolve_split_file(text_cls_path_cfg, split)
            if text_cls_file is None or text_cls_file in loaded_paths:
                continue
            loaded_paths.add(text_cls_file)
            text_cls_dict.update(np.load(text_cls_file, allow_pickle=True).item())
        return text_cls_dict

    def _get_annotation_text(self, annotation, strip=False):
        for key in self.annotation_text_keys:
            if key in annotation:
                text = annotation[key]
                if strip:
                    text = text.strip()
                return text
        raise KeyError(
            f"annotation missing text field. checked keys={self.annotation_text_keys}, annotation={annotation}"
        )

    def _estimate_num_clips(self, video_anno):
        if "num_clips" in video_anno:
            return (video_anno["num_clips"] + self.opt.data.downsample_rate - 1) // self.opt.data.downsample_rate

        num_frames = int(video_anno["num_frames"])
        clip_stride = max(int(self.clip_stride), 1)
        return int(np.ceil(num_frames / clip_stride))

    def _sanitize_annotation_db(self, anno_db):
        return anno_db

    def _parse_annotations(self, anno_file):
        with open(anno_file, "r") as f:
            anno = json.load(f)

        anno_db = {}
        for split in self.split:
            assert split in anno, f"split [{split}] does not exist"
            anno_db.update(anno[split])

        anno_db = self._sanitize_annotation_db(anno_db)

        dup_ct = 0
        vid_dict, text_dict = OrderedDict(), OrderedDict()

        for vid_id, value in anno_db.items():
            if "annotations" not in value:
                continue

            fps = float(value["fps"])
            num_frames = int(value["num_frames"])
            duration = float(value.get("duration", num_frames / fps))
            num_clips = self._estimate_num_clips(value)

            text_ids, texts, raw_texts, segments = [], [], [], []
            for text_idx, pair in enumerate(value["annotations"]):
                start = max(float(pair["segment"][0]), 0)
                end = min(float(pair["segment"][1]), duration)
                if end <= start:
                    continue

                raw_text = self._get_annotation_text(pair, strip=False)
                text = self._get_annotation_text(pair, strip=True)
                text_id = pair.get("sentence_id", f"{vid_id}_{text_idx:04d}")
                text_ids.append(text_id)
                texts.append(text)
                raw_texts.append(raw_text)
                segments.append((start, end))

                text_dict[text_id] = {
                    "text": text,
                    "segment": np.array((start, end))[None],
                    "text_idx": text_idx,
                    "vid_id": vid_id,
                }

            if len(text_ids) == 0:
                continue
            if len(set(text_ids)) < len(text_ids):
                dup_ct += 1

            vid_dict[vid_id] = {
                "fps": fps,
                "num_frames": num_frames,
                "num_clips": num_clips,
                "duration": duration,
                "text_ids": tuple(text_ids),
                "texts": tuple(texts),
                "raw_texts": tuple(raw_texts),
                "segments": np.array(segments),
            }

        print(f"duplicate text ids in {dup_ct:d} videos")

        # Keep old behavior to avoid storing all text twice.
        text_dict = None
        return vid_dict, text_dict

    def _parse_clip_info_file(self, fname):
        with open(fname, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            return data

        if isinstance(data, dict):
            rows = data.get("clips", [])
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"unsupported clip info format in {fname}")

        video_info = {}
        for row in rows:
            clip_id = row.get("clip_uid", row.get("clip_id", row.get("id")))
            if clip_id is None:
                continue

            info = dict(row)
            info.setdefault("video_uid", row.get("video_id", clip_id))
            info.setdefault("video_start_sec", row.get("start_sec", 0.0))
            video_info[clip_id] = info
        return video_info

    def _build_video_info(self):
        video_info = {}

        clip_info_path_cfg = self.opt.data.get("clip_info_files")
        if clip_info_path_cfg is not None:
            for split in self.split:
                clip_info_file = self._resolve_split_file(clip_info_path_cfg, split)
                if clip_info_file is None:
                    continue
                video_info.update(self._parse_clip_info_file(clip_info_file))
            if len(video_info) > 0:
                return video_info

        for clip_id, vid_meta in self.vid_dict.items():
            video_info[clip_id] = {
                "video_uid": clip_id,
                "video_start_sec": 0.0,
                "video_end_sec": float(vid_meta["duration"]),
            }
        return video_info

    def _get_video_info(self, clip_id):
        info = self.video_info.get(clip_id)
        if info is not None:
            return info

        if clip_id in self.vid_dict:
            return {
                "video_uid": clip_id,
                "video_start_sec": 0.0,
                "video_end_sec": float(self.vid_dict[clip_id]["duration"]),
            }

        raise KeyError(f"missing video mapping for clip {clip_id}")

    def _load_full_pt(self, path_stem):
        clip_id = os.path.basename(path_stem)
        folder = os.path.dirname(path_stem)
        info = self._get_video_info(clip_id)

        source_video = info.get("video_uid", clip_id)
        feature = torch.load(os.path.join(folder, source_video + ".pt")).numpy()

        start_sec = float(info.get("video_start_sec", info.get("start_sec", 0.0)))
        end_sec = info.get("video_end_sec", info.get("end_sec", None))
        if end_sec is None:
            return feature

        div = lambda x: int(float(x) * self.full_pt_fps // self.opt.data.clip_stride)
        start = max(0, div(start_sec))
        end = min(div(end_sec), feature.shape[0] + 1)
        return feature[start:end]

    def _read_video_frames(self, clip_id, frame_to_read):
        info = self._get_video_info(clip_id)

        video_id = info.get("video_uid", clip_id)
        video_name = self.video_fname_template.format(video_uid=video_id, clip_id=clip_id)
        if os.path.isabs(video_name):
            video_file = video_name
        else:
            video_file = os.path.join(self.opt.data.video_dir, video_name)

        video_reader = self._with_retry(decord.VideoReader, video_file)

        num_frames = len(video_reader)
        start_sec = float(info.get("video_start_sec", info.get("start_sec", 0.0)))
        start_offset = int(start_sec * self.decode_fps)

        frame_to_read = frame_to_read + start_offset
        frame_to_read = np.clip(frame_to_read, start_offset, num_frames - 1)
        if len(frame_to_read.shape) == 0:
            frame_to_read = [frame_to_read]

        max_attempts = 10
        for _ in range(max_attempts):
            try:
                frames = video_reader.get_batch(frame_to_read).permute(0, 3, 1, 2).float() / 255
                return self.transforms(frames)
            except Exception as e:
                print(f"Error loading {video_file}: {e}")

        raise RuntimeError(f"failed to decode frames from {video_file}")

    def _load_vid_feats(self, vid_id):
        if vid_id in self.vid_feat_dict:
            return self.vid_feat_dict[vid_id]

        vid_feat_files = [os.path.join(d, vid_id) for d in self.vid_feat_dir]
        vid_load_mode = self.opt.data.vid_load
        if vid_load_mode == "full_pt":
            vid_feats = [self._load_full_pt(f) for f in vid_feat_files]
        else:
            if vid_load_mode not in COMMON_VID_LOAD_FUNC:
                raise ValueError(f"unsupported vid_load mode: {vid_load_mode}")
            vid_feats = [COMMON_VID_LOAD_FUNC[vid_load_mode](f, self) for f in vid_feat_files]

        vid_feats = self._merge_video_feat_sources(vid_feats, vid_id, assert_aligned=True)
        vid_feats = self._to_video_tensor(vid_feats, self.downsample_rate)
        self.vid_feat_dict[vid_id] = vid_feats
        return vid_feats

    def _create_label(self):
        self.text2vid_label = defaultdict(list)
        self.vid2text_label = defaultdict(list)

        for vid_id, vid_dict in self.vid_dict.items():
            num_clips = vid_dict["num_clips"]
            f2i_start = self.f2i_start[:num_clips]
            f2i_end = self.f2i_end[:num_clips]

            for text, (start, end) in zip(vid_dict["raw_texts"], vid_dict["segments"]):
                start = int(start * vid_dict["fps"])
                end = int(end * vid_dict["fps"])
                feat_idx = np.where(np.logical_and(f2i_start <= end, f2i_end >= start))[0]
                self.text2vid_label[(vid_id, text)].append(feat_idx)
                for feat_id in feat_idx:
                    self.vid2text_label[(vid_id, feat_id)].append(text)

        dup_text_ct = 0
        for key, texts in self.vid2text_label.items():
            unique_texts = list(set(texts))
            self.vid2text_label[key] = unique_texts
            if len(unique_texts) > 1:
                dup_text_ct += 1

        print("Ratio of pos frames with multi-text", dup_text_ct, len(self.vid2text_label))

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        vid_feats = self._with_retry(self._load_vid_feats, vid_id)
        vid_len = vid_feats.size(1)
        clip_size, clip_stride = self.clip_size, self.clip_stride

        assert not self.to_fixed_len
        clip_offset = 0.5 * clip_size / clip_stride

        segments = np.clip(
            vid_dict["segments"][np.array(seg_idx)] * vid_dict["fps"],
            a_min=0,
            a_max=vid_dict["num_frames"],
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(np.ascontiguousarray(segments.astype(np.float32)))

        se = None
        if self.is_training:
            vid_feats, segments, se = self._truncate_vid_feats(vid_feats, segments, clip_offset)

        text_feats_list = tuple(self._with_retry(self._load_text_feats, vid_dict["text_ids"][i]) for i in seg_idx)

        feature_idx = np.arange(se[0], se[1]) if se is not None else np.arange(vid_len)
        texts = list(vid_dict["raw_texts"])
        text_label = {i: self.vid2text_label.get((vid_id, f), None) for i, f in enumerate(feature_idx)}
        text_cls_dict = {text: self.text_cls_dict[text][0] for text in texts}

        feature_idx = feature_idx[:: self.opt.encoder.fd]
        if self.opt.data.clip_d == 8:
            window = np.arange(0, self.clip_size, step=self.opt.data.clip_d) - (self.clip_size // 2)
        else:
            n = int(self.clip_size / self.opt.data.clip_d)
            window = np.linspace(0, self.clip_size, n, dtype=int) - (self.clip_size // 2)

        fidx2tlist_raw = (feature_idx[:, None] + clip_offset) * self.clip_stride + window
        tlist = np.unique(fidx2tlist_raw.flatten())

        tmap = {t: i for i, t in enumerate(tlist)}
        fidx2rel_tlist = np.array([tmap[x] for x in fidx2tlist_raw.reshape(-1)])
        fidx2rel_tlist = fidx2rel_tlist.reshape(fidx2tlist_raw.shape)
        fidx2rel_tlist = torch.from_numpy(fidx2rel_tlist)

        frames = self._read_video_frames(vid_id, tlist)

        assert self.ext_score_dir is None

        return {
            "fps": vid_dict["fps"],
            "num_frames": vid_dict["num_frames"],
            "duration": vid_dict["duration"],
            "segment": vid_dict["segments"],
            "clip_size": clip_size,
            "clip_stride": clip_stride,
            "target": segments,
            "clip_id": vid_id,
            "text_id": seg_idx,
            "vid": vid_feats,
            "text": text_feats_list,
            "frame": frames,
            "f2t": fidx2rel_tlist,
            "text_cls_dict": text_cls_dict,
            "text_label": text_label,
            "is_last": True,
        }


@register_dataset("ego4d_pretraining")
class Ego4DPretrainingDataset(PretrainingDataset):
    def __init__(
        self,
        split,
        is_training,
        anno_file=None,
        vid_feat_dir=None,
        text_feat_dir=None,
        ext_score_dir=None,
        tokenizer=None,
        clip_size=32,
        clip_stride=8,
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
        group_method="greedy",
        num_epochs=1,
        **kwargs,
    ):
        opt = kwargs.get("opt", None)
        ego4d_cfg = opt.data.get("ego4d", None) if opt is not None else None
        if ego4d_cfg is None or ego4d_cfg.get("anno_file") is None:
            raise ValueError("missing Ego4D annotation file. set data.ego4d.anno_file")
        anno_file = ego4d_cfg.get("anno_file")

        super().__init__(
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
            max_vid_len=max_vid_len,
            max_text_len=max_text_len,
            crop_ratio=crop_ratio,
            trunc_thresh=trunc_thresh,
            max_num_text=max_num_text,
            group_method=group_method,
            num_epochs=num_epochs,
            **kwargs,
        )

    def _sanitize_annotation_db(self, anno_db):
        if EGO4D_BAD_VIDEO_ID not in anno_db:
            return anno_db

        filtered = []
        for ann in anno_db[EGO4D_BAD_VIDEO_ID].get("annotations", []):
            text = self._get_annotation_text(ann, strip=True)
            if text != EGO4D_BAD_SENTENCE:
                filtered.append(ann)
        anno_db[EGO4D_BAD_VIDEO_ID]["annotations"] = filtered
        return anno_db

    def _parse_ego4d_video_info(self, ego4d_anno, ego4d_meta):
        with open(ego4d_meta, "r") as f:
            metadata = json.load(f)
        v2meta = {v["video_uid"]: v["video_metadata"] for v in metadata["videos"]}

        with open(ego4d_anno, "r") as f:
            anno = json.load(f)

        video_info = {}
        for video in anno["videos"]:
            for clip in video["clips"]:
                info = {
                    "video_start_sec": clip["video_start_sec"],
                    "video_end_sec": clip["video_end_sec"],
                    "video_uid": video["video_uid"],
                }

                if video["video_uid"].startswith("grp"):
                    h, w = 1440, 1920
                else:
                    meta = v2meta[video["video_uid"]]
                    h = meta["display_resolution_height"]
                    w = meta["display_resolution_width"]
                info["height"] = h
                info["width"] = w

                video_info[clip["clip_uid"]] = info

        return video_info

    def _build_video_info(self):
        ego4d_cfg = self.opt.data.get("ego4d")
        metadata_file = ego4d_cfg.get("metadata")

        video_info = {}
        for split in self.split:
            clip_anno_file = ego4d_cfg.get(f"clip_{split}_anno")
            if clip_anno_file is None:
                continue
            video_info.update(self._parse_ego4d_video_info(clip_anno_file, metadata_file))

        return video_info


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
