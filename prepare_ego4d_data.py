import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


TASK_CONFIG = {
    "nlq": {
        "annotation_names": {"train": "nlq_train.json", "val": "nlq_val.json"},
        "official_fallbacks": {
            "train": "/tmp/decafnet/data/datasets/ego4d_nlq_train.json",
            "val": "/tmp/decafnet/data/datasets/ego4d_nlq_val.json",
        },
        "video_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/v1/video/32x8_d2",
        "text_cls_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/v1/text_cls",
        "text_768_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/v1/text/text_768d",
        "annotation_json": "/tmp/decafnet/data/datasets/egovlpv1/features/v1/nlq_32x8_d2.json",
        "clip_token_path": "/tmp/decafnet/data/datasets/clip/nlq_token_features_B_32.npy",
    },
    "goalstep": {
        "annotation_names": {"train": "train.json", "val": "val.json"},
        "official_fallbacks": {
            "train": "/tmp/decafnet/data/datasets/goalstep_annotations/train.json",
            "val": "/tmp/decafnet/data/datasets/goalstep_annotations/val.json",
        },
        "video_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/goalstep/video/32x8_d2",
        "text_cls_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/goalstep/text_cls",
        "text_768_dir": "/tmp/decafnet/data/datasets/egovlpv1/features/goalstep/text/text_768d",
        "annotation_json": "/tmp/decafnet/data/datasets/egovlpv1/features/goalstep/32x8_d2.json",
        "clip_token_path": "/tmp/decafnet/data/datasets/clip/goalstep_token_features_B_32.npy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Ego4D text/video artifacts consumed by the DeCafNet release. "
            "Supports Ego4D-NLQ and GoalStep."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        default="nlq",
        choices=sorted(TASK_CONFIG),
        help="Ego4D task to prepare: nlq or goalstep.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="video,text_cls,text_768d,annotations,clip_token",
        help=(
            "Comma-separated steps to run. "
            "Supported: video,text_cls,text_768d,annotations,clip_token"
        ),
    )
    parser.add_argument(
        "--egovlp-python",
        type=str,
        default=sys.executable,
        help="Python executable used to run egovlp/run/test_nlq.py",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to EgoVLP checkpoint used for feature extraction",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA device id forwarded to egovlp/run/test_nlq.py",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="Directory containing raw Ego4D videos as <video_uid>.mp4",
    )
    parser.add_argument(
        "--metadir",
        type=str,
        default=None,
        help=(
            "Directory containing official annotations. "
            "For nlq: nlq_train.json / nlq_val.json. "
            "For goalstep: train.json / val.json."
        ),
    )
    parser.add_argument(
        "--official-train-json",
        type=str,
        default=None,
        help="Explicit path to official train JSON for packaging.",
    )
    parser.add_argument(
        "--official-val-json",
        type=str,
        default=None,
        help="Explicit path to official val JSON for packaging.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Output directory for EgoVLP video features",
    )
    parser.add_argument(
        "--text-cls-dir",
        type=str,
        default=None,
        help="Output directory for sentence_256_{split}.npy",
    )
    parser.add_argument(
        "--text-768-dir",
        type=str,
        default=None,
        help="Output directory for packaged text/text_768d/<sentence_id>.npy",
    )
    parser.add_argument(
        "--annotation-json",
        type=str,
        default=None,
        help="Output packaged annotation JSON",
    )
    parser.add_argument(
        "--clip-token-path",
        type=str,
        default=None,
        help="Output CLIP token feature dictionary",
    )
    parser.add_argument(
        "--window-d",
        type=int,
        default=2,
        help="Temporal dilation d used for 32x8_d{d} video features",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1000000000,
        help="Max metadata rows passed to egovlp/run/test_nlq.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping them",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Temporary directory for intermediate text extraction outputs",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def require_path(value, flag, reason):
    if value:
        return value
    raise ValueError(f"{flag} is required when running {reason}.")


def apply_task_defaults(args):
    task_cfg = TASK_CONFIG[args.task]
    if args.video_dir is None:
        args.video_dir = task_cfg["video_dir"]
    if args.text_cls_dir is None:
        args.text_cls_dir = task_cfg["text_cls_dir"]
    if args.text_768_dir is None:
        args.text_768_dir = task_cfg["text_768_dir"]
    if args.annotation_json is None:
        args.annotation_json = task_cfg["annotation_json"]
    if args.clip_token_path is None:
        args.clip_token_path = task_cfg["clip_token_path"]
    return args


def resolve_official_json(args, split):
    explicit = getattr(args, f"official_{split}_json")
    if explicit:
        return explicit

    task_cfg = TASK_CONFIG[args.task]
    if args.metadir:
        return os.path.join(args.metadir, task_cfg["annotation_names"][split])

    fallback = task_cfg["official_fallbacks"][split]
    if os.path.exists(fallback):
        return fallback

    raise ValueError(
        f"Could not resolve official {args.task} {split} annotations. "
        f"Pass --official-{split}-json or --metadir."
    )


def validate_args(args, steps):
    extraction_steps = {"video", "text_cls", "text_768d"}
    if steps & extraction_steps:
        reason = ",".join(sorted(steps & extraction_steps))
        require_path(args.resume, "--resume", reason)
        require_path(args.video_root, "--video-root", reason)
        require_path(args.metadir, "--metadir", reason)

    if "annotations" in steps:
        resolve_official_json(args, "train")
        resolve_official_json(args, "val")


def parse_official_ego4d_queries(path):
    with open(path) as fp:
        data = json.load(fp)

    clip_info = OrderedDict()
    for video in data["videos"]:
        for clip in video["clips"]:
            queries = []
            for ann in clip["annotations"]:
                queries.extend(ann["language_queries"])
            clip_info[clip["clip_uid"]] = {
                "video_uid": video["video_uid"],
                "video_start_sec": clip["video_start_sec"],
                "video_end_sec": clip["video_end_sec"],
                "annotations": queries,
            }
    return clip_info


def load_pickle_dict(path):
    return np.load(path, allow_pickle=True).item()


def build_text_id_maps(text_cls_train_path, text_cls_val_path):
    train_dict = load_pickle_dict(text_cls_train_path)
    val_dict = load_pickle_dict(text_cls_val_path)
    train_text2id = {text: idx for idx, text in enumerate(train_dict.keys())}
    val_offset = len(train_text2id)
    val_text2id = {text: val_offset + idx for idx, text in enumerate(val_dict.keys())}
    return train_dict, val_dict, train_text2id, val_text2id


def package_annotations(official_json, text2id, video_dir):
    official = parse_official_ego4d_queries(official_json)
    packaged = OrderedDict()

    for clip_id, clip in official.items():
        duration = clip["video_end_sec"] - clip["video_start_sec"]
        annotations = []
        for q in clip["annotations"]:
            text = q.get("query")
            if text is None:
                continue
            start = q["clip_start_sec"]
            end = q["clip_end_sec"]
            if end - start < 0.25:
                continue
            if text not in text2id:
                continue
            annotations.append(
                {
                    "sentence": text,
                    "segment": [start, end],
                    "sentence_id": text2id[text],
                }
            )

        if not annotations:
            continue

        video_feature = np.load(os.path.join(video_dir, clip_id + ".npy"))
        packaged[clip_id] = {
            "duration": duration,
            "fps": 30,
            "annotations": annotations,
            "num_frames": int(duration * 30),
            "num_clips": int(video_feature.shape[0]),
        }

    return packaged


def run_subprocess(cmd, cwd, env):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_egovlp_extraction(
    egovlp_dir,
    python_bin,
    resume,
    gpu,
    video_root,
    metadir,
    task,
    split,
    subsample,
    save_feats,
    max_items,
    d=None,
    token=False,
):
    ensure_dir(save_feats)
    env = os.environ.copy()
    project_root = str(Path(egovlp_dir).resolve().parent)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        python_bin,
        "run/test_nlq.py",
        "--split",
        split,
        "--subsample",
        subsample,
        "--resume",
        resume,
        "--gpu",
        str(gpu),
        "--data",
        video_root,
        "--metadir",
        metadir,
        "--task",
        task,
        "--s",
        "0",
        "--l",
        str(max_items),
        "--save_feats",
        save_feats,
    ]
    if d is not None:
        cmd.extend(["--d", str(d)])
    if token:
        cmd.extend(["--token", "True"])

    run_subprocess(cmd, cwd=egovlp_dir, env=env)


def maybe_extract_video(args, egovlp_dir):
    video_dir = Path(args.video_dir)
    if video_dir.exists() and any(video_dir.glob("*.npy")) and not args.overwrite:
        print(f"Skip video extraction, found existing files in {video_dir}")
        return

    ensure_dir(video_dir)
    for split in ("train", "val"):
        run_egovlp_extraction(
            egovlp_dir=egovlp_dir,
            python_bin=args.egovlp_python,
            resume=args.resume,
            gpu=args.gpu,
            video_root=args.video_root,
            metadir=args.metadir,
            task=args.task,
            split=split,
            subsample="video",
            save_feats=str(video_dir),
            max_items=args.max_items,
            d=args.window_d,
            token=False,
        )


def maybe_extract_text_cls(args, egovlp_dir):
    text_cls_dir = Path(args.text_cls_dir)
    ensure_dir(text_cls_dir)
    expected = [
        text_cls_dir / "sentence_256_train.npy",
        text_cls_dir / "sentence_256_val.npy",
    ]
    if all(p.exists() for p in expected) and not args.overwrite:
        print(f"Skip text_cls extraction, found {expected[0].name} and {expected[1].name}")
        return

    scratch_root = Path(args.scratch_dir) if args.scratch_dir else Path(
        tempfile.mkdtemp(prefix=f"egovlp_{args.task}_text_cls_")
    )
    ensure_dir(scratch_root)

    try:
        for split in ("train", "val"):
            split_dir = scratch_root / "text_cls" / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            ensure_dir(split_dir)
            run_egovlp_extraction(
                egovlp_dir=egovlp_dir,
                python_bin=args.egovlp_python,
                resume=args.resume,
                gpu=args.gpu,
                video_root=args.video_root,
                metadir=args.metadir,
                task=args.task,
                split=split,
                subsample="text",
                save_feats=str(split_dir),
                max_items=args.max_items,
                d=None,
                token=False,
            )
            src = split_dir / f"sentence_256_{split}.npy"
            dst = text_cls_dir / src.name
            shutil.move(str(src), str(dst))
            print(f"Wrote {dst}")
    finally:
        if args.scratch_dir is None and scratch_root.exists():
            shutil.rmtree(scratch_root)


def package_text_768_features(text_768_train_path, text_768_val_path, text_cls_dir, out_dir):
    train_raw = load_pickle_dict(text_768_train_path)
    val_raw = load_pickle_dict(text_768_val_path)
    _, _, train_text2id, val_text2id = build_text_id_maps(
        Path(text_cls_dir) / "sentence_256_train.npy",
        Path(text_cls_dir) / "sentence_256_val.npy",
    )

    ensure_dir(out_dir)
    for text, sent_id in train_text2id.items():
        np.save(os.path.join(out_dir, f"{sent_id}.npy"), train_raw[text])
    for text, sent_id in val_text2id.items():
        np.save(os.path.join(out_dir, f"{sent_id}.npy"), val_raw[text])


def maybe_extract_text_768d(args, egovlp_dir):
    text_768_dir = Path(args.text_768_dir)
    if text_768_dir.exists() and any(text_768_dir.glob("*.npy")) and not args.overwrite:
        print(f"Skip text_768d export, found existing files in {text_768_dir}")
        return

    scratch_root = Path(args.scratch_dir) if args.scratch_dir else Path(
        tempfile.mkdtemp(prefix=f"egovlp_{args.task}_text_768_")
    )
    ensure_dir(scratch_root)

    try:
        for split in ("train", "val"):
            split_dir = scratch_root / "text_768_raw" / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            ensure_dir(split_dir)
            run_egovlp_extraction(
                egovlp_dir=egovlp_dir,
                python_bin=args.egovlp_python,
                resume=args.resume,
                gpu=args.gpu,
                video_root=args.video_root,
                metadir=args.metadir,
                task=args.task,
                split=split,
                subsample="text",
                save_feats=str(split_dir),
                max_items=args.max_items,
                d=None,
                token=True,
            )

        package_text_768_features(
            text_768_train_path=scratch_root / "text_768_raw" / "train" / "sentence_768_train.npy",
            text_768_val_path=scratch_root / "text_768_raw" / "val" / "sentence_768_val.npy",
            text_cls_dir=args.text_cls_dir,
            out_dir=args.text_768_dir,
        )
        print(f"Wrote packaged text_768d to {args.text_768_dir}")
    finally:
        if args.scratch_dir is None and scratch_root.exists():
            shutil.rmtree(scratch_root)


def load_openai_clip_backend(model_name="ViT-B/32", device="cpu"):
    try:
        import clip
        import torch
    except Exception as exc:
        raise RuntimeError(
            "CLIP token generation requires the OpenAI CLIP package. "
            "Install it in the active env, for example: "
            "`pip install git+https://github.com/openai/CLIP.git`."
        ) from exc

    model, _ = clip.load(model_name, device=device)
    model.eval()

    def encode(texts):
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            x = model.token_embedding(tokens).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)
            x = x @ model.text_projection

        outputs = []
        for i in range(tokens.shape[0]):
            eot_idx = int(tokens[i].argmax())
            token_feat = x[i, 1:eot_idx].float().cpu().numpy().astype(np.float32)
            outputs.append(token_feat)
        return outputs

    return encode


def maybe_generate_clip_tokens(args):
    clip_token_path = Path(args.clip_token_path)
    if clip_token_path.exists() and not args.overwrite:
        print(f"Skip CLIP token export, found {clip_token_path}")
        return

    train_cls, val_cls, _, _ = build_text_id_maps(
        Path(args.text_cls_dir) / "sentence_256_train.npy",
        Path(args.text_cls_dir) / "sentence_256_val.npy",
    )
    unique_texts = list(train_cls.keys()) + [text for text in val_cls.keys() if text not in train_cls]

    encode = load_openai_clip_backend(device="cpu")
    outputs = {}
    batch_size = 64
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start:start + batch_size]
        feats = encode(batch)
        for text, feat in zip(batch, feats):
            outputs[text] = feat
        print(f"Encoded CLIP token features {start + len(batch)}/{len(unique_texts)}")

    ensure_dir(clip_token_path.parent)
    np.save(str(clip_token_path), outputs)
    print(f"Wrote {clip_token_path}")


def maybe_package_annotations(args):
    _, _, train_text2id, val_text2id = build_text_id_maps(
        Path(args.text_cls_dir) / "sentence_256_train.npy",
        Path(args.text_cls_dir) / "sentence_256_val.npy",
    )

    official_train = resolve_official_json(args, "train")
    official_val = resolve_official_json(args, "val")

    data = {
        "train": package_annotations(
            official_json=official_train,
            text2id=train_text2id,
            video_dir=args.video_dir,
        ),
        "val": package_annotations(
            official_json=official_val,
            text2id=val_text2id,
            video_dir=args.video_dir,
        ),
    }

    out_path = Path(args.annotation_json)
    ensure_dir(out_path.parent)
    with open(out_path, "w") as fp:
        json.dump(data, fp)
    print(f"Wrote {out_path}")


def main():
    args = apply_task_defaults(parse_args())
    steps = {step.strip() for step in args.steps.split(",") if step.strip()}
    valid_steps = {"video", "text_cls", "text_768d", "annotations", "clip_token"}
    invalid = sorted(steps - valid_steps)
    if invalid:
        raise ValueError(f"Unsupported steps: {invalid}")
    validate_args(args, steps)

    egovlp_dir = Path(__file__).resolve().parent

    if "video" in steps:
        maybe_extract_video(args, egovlp_dir)
    if "text_cls" in steps:
        maybe_extract_text_cls(args, egovlp_dir)
    if "text_768d" in steps:
        maybe_extract_text_768d(args, egovlp_dir)
    if "annotations" in steps:
        maybe_package_annotations(args)
    if "clip_token" in steps:
        maybe_generate_clip_tokens(args)


if __name__ == "__main__":
    main()
