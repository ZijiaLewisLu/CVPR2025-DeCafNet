import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torchtext
from yacs.config import CfgNode

from libs.core.opt import _update_eval_data, _update_opt, get_cfg_defaults
from libs.utils.dist_utils import get_local_rank
from libs.utils.opt_utils import update_from


try:
    torchtext.disable_torchtext_deprecation_warning()
except Exception:
    pass


def main(args):
    # Resolve checkpoint path/stem and experiment root.

    ckpt_arg = args.ckpt
    ckpt_file = os.path.abspath(ckpt_arg)

    ckpt_models_dir = os.path.dirname(ckpt_file)
    root = os.path.dirname(ckpt_models_dir)
    ckpt = os.path.splitext(os.path.basename(ckpt_file))[0]

    # Load and refresh config.
    opt_path = os.path.join(root, "opt.yaml")
    if not os.path.isfile(opt_path):
        raise FileNotFoundError(f"missing opt.yaml: {opt_path}")
    with open(opt_path, "r") as f:
        opt = CfgNode.load_cfg(f)

    task = opt.get("task", None)
    if task not in {"svg", "ego4d", "mad"}:
        raise ValueError(f"unsupported task for eval: {task}")
    opt = update_from(opt, get_cfg_defaults(task))
    if args.set_cfgs:
        opt.merge_from_list(args.set_cfgs)

    _update_opt(opt)
    _update_eval_data(opt)

    opt.aux.logdir = root
    opt.aux.dryrun = args.dryrun
    opt["_root"] = root
    opt["_ckpt"] = ckpt

    # Runtime setup.
    torch.cuda.set_device(get_local_rank())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"Evaluating task={opt.task} root={root} ckpt={ckpt}")

    if opt.task == "svg":
        from libs.svg.worker import Evaluator
    elif opt.task == "ego4d":
        from libs.ego4d.worker import Evaluator
    elif opt.task == "mad":
        from libs.mad.worker import Evaluator

    evaluator = Evaluator(opt)
    evaluator.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="last",
        help="checkpoint stem (e.g., last, 10-7800) or path to a .pth file",
    )
    parser.add_argument("--dryrun", action="store_true", help="evaluate only a few batches")
    #     "--save-ckpt",
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        nargs=argparse.REMAINDER,
        default=[],
        help="optional cfg overrides, same format as train.py",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    main(parser.parse_args())
