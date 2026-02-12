import math
import os
import random
from copy import deepcopy
from functools import partial
import pickle as pk

import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtext.data import get_tokenizer
from tqdm import trange

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id, num_workers, rank):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31   # base_seed + worker_id
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)


def worker_init_reset_seed_correct(num_workers, rank, worker_id):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31   # base_seed + worker_id
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)


tokenizers = dict()


def register_tokenizer(name):
    def decorator(module):
        tokenizers[name] = module
        return module
    return decorator


@register_tokenizer('glove')
class GloVeTokenizer:

    def __init__(self, name='6B'):
        torchtext.vocab.GloVe.url['6B'] = (
            'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip'
        )
        self.vocab = torchtext.vocab.GloVe(name=name)
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, text, max_len=None):
        words = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
        if max_len is not None:
            feats = feats[:max_len]
        feats = feats.transpose(0, 1)   # (c, t)
        return feats


def make_tokenizer(name):
    return tokenizers[name]()


def create_dataset_registry():
    datasets = dict()

    def register_dataset(name):
        def decorator(module):
            datasets[name] = module
            return module
        return decorator

    return datasets, register_dataset


def load_pk(fname, n=0):
    with open(fname, 'rb') as f:
        return pk.load(f)[n]


def load_pk_avg(fname):
    with open(fname, 'rb') as f:
        v_align, v_distill = pk.load(f)[:2]
    return (v_align + v_distill) / 2


VID_LOAD_FUNC = {
    "npy": lambda x, d: np.load(x + '.npy').astype(np.float32),
    "pk1": lambda x, d: load_pk(x + '.pk', 1),
    "pk0": lambda x, d: load_pk(x + '.pk', 0),
    "pk_avg": lambda x, d: load_pk_avg(x + '.pk'),
    "pt": lambda x, d: torch.load(x + '.pt').numpy(),
}


def make_dataset_from_registry(opt, datasets, make_tokenizer_fn, num_epochs=1, is_training=True):
    opt = deepcopy(opt)
    if not is_training:
        opt.data = opt.eval.data
    data_opt = opt.data

    if 'tokenizer' in data_opt:
        tokenizer = make_tokenizer_fn(data_opt.pop('tokenizer'))
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
        opt=opt,
        tokenizer=tokenizer,
        is_training=is_training,
        num_epochs=num_epochs,
        **data_opt,
    )


def make_dataloader_common(
    dataset,
    generator,
    batch_size,
    num_workers,
    is_training,
    world_size=1,
    rank=0,
):
    sampler = None
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
    )
    return loader, sampler





class BaseGroundingDataset(Dataset):
    def _setup_common(
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
        downsample_rate,
        to_fixed_len,
        normalize_vid,
        normalize_text,
        normalize_scores,
        temperature,
        max_vid_len,
        max_text_len,
        crop_ratio,
        trunc_thresh,
        max_num_text,
        group_method,
        num_epochs,
        require_text_feat_if_no_tokenizer=True,
    ):
        if isinstance(split, str) and "," in split:
            split = split.split(",")
        print(split)
        if not isinstance(split, (list, tuple)):
            split = (split,)
        if not isinstance(vid_feat_dir, (list, tuple)):
            vid_feat_dir = (vid_feat_dir,)

        if require_text_feat_if_no_tokenizer and tokenizer is None:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0

        self.anno_file = anno_file
        self.vid_feat_dir = list(vid_feat_dir)
        self.text_feat_dir = text_feat_dir
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

        self.group_method = group_method
        self.num_epochs = num_epochs

        self.vid_feat_dict = {}
        self.text_feat_dict = {}

    def _build_data_list(self, assert_eval_single_epoch=True):
        if self.is_training:
            self.data_list = self._build_train_samples()
            return
        if assert_eval_single_epoch:
            assert self.num_epochs == 1
        self.data_list = self._build_eval_samples()

    def _merge_video_feat_sources(self, vid_feats, vid_id, assert_aligned=False):
        if len(vid_feats) == 1:
            return vid_feats[0]

        feat_lens = [len(x) for x in vid_feats]
        max_len, min_len = max(feat_lens), min(feat_lens)
        if assert_aligned:
            assert max_len - min_len <= 10, (
                "misaligned features ([max] {:d}, [min] {:d}) for video {:s}".format(
                    max_len,
                    min_len,
                    vid_id,
                )
            )

        for idx in range(len(vid_feats)):
            if feat_lens[idx] < max_len:
                pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                vid_feats[idx] = np.concatenate((vid_feats[idx], pad))
        return np.concatenate(vid_feats, axis=-1)

    def _to_video_tensor(self, vid_feats, downsample_rate):
        if downsample_rate > 1:
            vid_feats = vid_feats[::downsample_rate]
        vid_feats = vid_feats.transpose()
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)
        return vid_feats

    def _truncate_vid_feats(self, feats, segments, offset, num_trials=5000):
        vid_len = feats.size(1)
        max_vid_len = self.max_vid_len

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, segments, None

            max_vid_len = random.randint(
                int(max(math.ceil(self.crop_ratio[0] * vid_len), 1)),
                int(min(math.ceil(self.crop_ratio[1] * vid_len), vid_len)),
            )
            if max_vid_len == vid_len:
                return feats, segments, None

        s0 = int(max(0, math.floor(segments[:, 0].max() - max_vid_len)))
        s1 = int(min(vid_len - max_vid_len, math.ceil(segments[:, 1].min())))

        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)
        if seg_lens.max() > (self.max_vid_len / self.trunc_thresh):
            trunc_thresh = 0.2
            print(
                f"data is too long {seg_lens.max()}, "
                f"change truncate threshold to {trunc_thresh}"
            )
        else:
            trunc_thresh = self.trunc_thresh

        for _ in range(num_trials):
            ws = random.randint(s0, s1)
            we = ws + max_vid_len

            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > trunc_thresh):
                feats = feats[:, ws:we]
                segments = torch.clamp(
                    segments - ws,
                    min=-offset,
                    max=we - ws + offset,
                )
                return feats, segments, [ws, we]
        raise ValueError("no valid truncation found")

    def _load_text_feats(self, text_id):
        if text_id in self.text_feat_dict:
            return self.text_feat_dict[text_id]

        if hasattr(self, "raw_text_feat_dict") and hasattr(self, "text_id2text"):
            text_feats = self.raw_text_feat_dict[self.text_id2text[text_id]]
            text_feats = text_feats.transpose()
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
        elif self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]["text"])
        else:
            try:
                text_feat_file = os.path.join(self.text_feat_dir, str(text_id) + ".npy")
                text_feats = np.load(text_feat_file).astype(np.float32)
            except Exception as exc:
                raise ValueError(
                    f"failed to load features for sentence {text_id}"
                ) from exc
            text_feats = text_feats.transpose()
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))

        if self.is_training and self.max_text_len:
            text_feats = text_feats[:, :self.max_text_len]
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)
        return text_feats

    def _load_ext_scores(self, text_id):
        try:
            score_file = os.path.join(self.ext_score_dir, text_id + ".npy")
            scores = np.load(score_file).astype(np.float32)
        except Exception as exc:
            raise ValueError(
                "failed to load external scores for sentence {:s}".format(text_id)
            ) from exc

        if self.downsample_rate > 1:
            scores = scores[::self.downsample_rate]
        scores = torch.from_numpy(np.ascontiguousarray(scores))[None]
        if self.normalize_scores:
            scores = torch.sigmoid(scores / self.temperature)

        self.text_feat_dict[text_id] = scores
        return scores

    def _avgpool_to_fixed_len(self, feats, size):
        vid_len = feats.size(1)
        sampling_ratio = math.ceil(vid_len / size)
        feats = F.interpolate(
            feats[None],
            size=size * sampling_ratio,
            mode="linear",
            align_corners=False,
        )
        if sampling_ratio > 1:
            feats = F.avg_pool1d(feats, kernel_size=sampling_ratio)
        return feats[0]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_train_samples(self):
        samples = []
        for _ in trange(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                samples.extend(self._group(vid_id))
        samples = samples[: len(samples) // self.num_epochs * self.num_epochs]
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples.extend([(vid_id, tuple(range(len(vid_dict["segments"]))))])
        return tuple(samples)

    def _group(self, vid_id):
        if self.to_fixed_len:
            return self._group_with_fixed_len(vid_id)
        return self._group_with_max_len(vid_id)

    def _group_with_fixed_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]
        idx = list(range(len(vid_dict["segments"])))
        if self.group_method in ("random", "all"):
            return [(vid_id, tuple(idx))]

        max_num_text = self.max_num_text or len(idx)
        random.shuffle(idx)
        samples = []
        for i in range(0, len(idx), max_num_text):
            sample = (vid_id, tuple(idx[i : i + max_num_text]))
            samples.append(sample)
        return samples

    def _group_with_max_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]
        num_clips = vid_dict.get("num_clips", None)

        if num_clips is not None and num_clips <= self.max_vid_len:
            win_len = num_clips
            if self.crop_ratio is not None:
                win_len = int(max(math.ceil(self.crop_ratio[0] * win_len), 1))
        else:
            win_len = self.max_vid_len

        win_len = (
            self.clip_stride * (win_len - 1) + self.clip_size
        ) / vid_dict["fps"]

        sort_idx = np.argsort(vid_dict["segments"][:, 0])
        segments = vid_dict["segments"][sort_idx]
        mask = np.ones(len(segments), dtype=bool)
        max_num_text = self.max_num_text or len(segments)

        samples = []
        while mask.sum() > 0:
            ptr = np.nonzero(mask)[0].min()

            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )
                idx = np.nonzero(is_inside)[0]
                if len(idx) > max_num_text:
                    idx = np.random.choice(idx, max_num_text, replace=False)
            samples.append((vid_id, tuple(sort_idx[idx])))
            mask[idx] = 0
        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def _with_retry(self, fn, *args, max_retry=10):
        for i in range(max_retry):
            try:
                return fn(*args)
            except Exception as exc:
                if i == (max_retry - 1):
                    raise exc
        raise RuntimeError("unreachable")

    def _prepare_video_item(self, vid_feats):
        vid_len = vid_feats.size(1)
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len and self.max_vid_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride
        return vid_feats, vid_len, clip_size, clip_stride, clip_offset

    def _compute_target_segments(self, vid_dict, seg_idx, clip_stride, clip_offset):
        segments = np.clip(
            vid_dict["segments"][np.array(seg_idx)] * vid_dict["fps"],
            a_min=0,
            a_max=vid_dict["num_frames"],
        ) / clip_stride - clip_offset
        return torch.from_numpy(np.ascontiguousarray(segments.astype(np.float32)))

    def _maybe_truncate_and_sample(self, vid_feats, segments, clip_offset, seg_idx):
        se = None
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments, se = self._truncate_vid_feats(
                    vid_feats,
                    segments,
                    clip_offset,
                )
            max_num_text = self.max_num_text or len(seg_idx)
            if self.group_method == "random" and len(seg_idx) > max_num_text:
                seg_idx = random.sample(list(seg_idx), k=max_num_text)
                segments = segments[seg_idx]
        return vid_feats, segments, seg_idx, se

    def _load_text_feature_list(self, vid_dict, seg_idx):
        text_feats_list = []
        for idx in seg_idx:
            text_feats = self._with_retry(
                self._load_text_feats,
                vid_dict["text_ids"][idx],
            )
            text_feats_list.append(text_feats)
        return tuple(text_feats_list)

    def _load_ext_scores_list(self, vid_dict, seg_idx):
        if self.is_training or self.ext_score_dir is None:
            return None

        ext_scores_list = []
        for idx in seg_idx:
            scores = self._load_ext_scores(vid_dict["text_ids"][idx])
            if self.to_fixed_len and self.max_vid_len:
                scores = self._avgpool_to_fixed_len(scores, self.max_vid_len)
            ext_scores_list.append(scores)
        if not ext_scores_list:
            return None
        return torch.cat(ext_scores_list)
