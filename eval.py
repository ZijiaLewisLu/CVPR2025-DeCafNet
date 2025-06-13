import argparse
import os
import torch
import torchtext
try:
    torchtext.disable_torchtext_deprecation_warning()
except AttributeError:
    pass

# from libs import Evaluator
# from libs.core.opt import _update_opt
import yaml
from yacs.config import CfgNode
# from libs.helper.azure import set_azure_status, mount, test_azcopy
from libs.helper.utils import create_wandb
from libs.core.utils import update_from, get_cfg_defaults
import math



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ckpt', type=str, default='last', help="checkpoint name")
    parser.add_argument('--dryrun', default=False, action='store_true')
    args = parser.parse_args()

    root = args.name
    with open(os.path.join(root, 'opt.yaml'), 'r') as fd:  
        opt = CfgNode.load_cfg(fd)
    task = opt.get('task', 'grounder')
    opt = update_from(opt, get_cfg_defaults(task))

    opt.aux.dryrun = args.dryrun
    opt['_root'] = root
    opt['_ckpt'] = args.ckpt

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    print(args.name, args.ckpt)

    if task == 'grounder':
        from libs.worker_v2 import Evaluator
        opt.aux.mount = '/tmp/zijia-2024-east'
        opt.data.shallow_ds = 1
        opt.eval.data.shallow_ds = 1
        # print(opt.eval.data.shallow_vid_feat_dir[0])
        # import ipdb; ipdb.set_trace()
        evaluator = Evaluator(opt)
        # evaluator.run(save_ckpt=True)
        evaluator.run()
        # print(opt.)
