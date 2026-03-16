import argparse
import os

import torch
from yacs.config import CfgNode

from libs.core.opt import _update_eval_data, _update_opt, get_cfg_defaults
from libs.utils.dist_utils import get_local_rank, get_rank, get_world_size, print0
from libs.utils.opt_utils import update_from


def load_opt(root, split, dryrun):
    opt_path = os.path.join(root, "opt.yaml")
    if not os.path.isfile(opt_path):
        raise FileNotFoundError(f"missing opt.yaml: {opt_path}")
    with open(opt_path, "r") as fd:
        opt = CfgNode.load_cfg(fd)

    task = opt.get("task", None)
    if task != "encoder":
        raise ValueError(f"extract_sidekick_feature.py expects task=encoder, got {task}")

    opt = update_from(opt, get_cfg_defaults(task))
    _update_opt(opt)
    _update_eval_data(opt)

    opt.aux.dryrun = dryrun
    opt.data.eval_split = split
    opt.eval.data.split = split

    opt["_root"] = root
    opt["_distributed"] = get_world_size() > 1
    opt["_world_size"] = get_world_size()
    return opt


def main(args):
    root = os.path.abspath(args.root)
    ckpt_name = args.ckpt
    ckpt_path = os.path.join(root, "models", f"{ckpt_name}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    torch.cuda.set_device(get_local_rank())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    opt = load_opt(root, args.split, args.dryrun)
    opt["_ckpt"] = ckpt_name

    from libs.encoder.worker import Evaluator
    from libs.encoder.model import SidekickEncoder

    evaluator = Evaluator(opt)
    model = SidekickEncoder(opt).cuda().eval().requires_grad_(False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.vid_model._feature_extract = args.feature_extract_batch

    save_root = os.path.abspath(args.output) if args.output else os.path.join(root, "features")
    save_dir = os.path.join(save_root, ckpt_name)
    os.makedirs(save_dir, exist_ok=True)

    model_data = {
        "epoch": ckpt.get("epoch", 0),
        "iteration": ckpt.get("itr", 0),
        "model": model,
        "opt": opt,
        "save_fdr": save_dir,
    }
    model_dict = {0: model_data}

    print0(
        f"Extract sidekick feature | rank={get_rank()}/{get_world_size()} "
        f"split={args.split} ckpt={ckpt_name} save_dir={save_dir}"
    )
    evaluator._run_feature_extraction(model_dict, wandb_run=None, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="experiment log directory")
    parser.add_argument("--ckpt", type=str, default="last", help="checkpoint stem under <root>/models")
    parser.add_argument("--split", type=str, default="val", help="dataset split for extraction")
    parser.add_argument("--output", type=str, default=None, help="optional output root directory")
    parser.add_argument("--feature-extract-batch", type=int, default=512)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    main(parser.parse_args())
