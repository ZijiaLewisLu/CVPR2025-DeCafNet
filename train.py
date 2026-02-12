import argparse
import os

import torchtext

try:
    torchtext.disable_torchtext_deprecation_warning()
except:
    pass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from libs.utils.opt_utils import setup_cfg, create_wandb
from libs.utils.dist_utils import print0, get_rank, barrier, get_local_rank
from datetime import timedelta

def main(rank, args): # opt, wandb_run):
    print(f"Training process: {rank} {get_local_rank()}")
    if len(args.cfg_file) == 1:
        args.cfg_file = args.cfg_file[0].split(' ')
    if len(args.set_cfgs) == 1:
        args.set_cfgs = args.set_cfgs[0].split(' ')
    opt = setup_cfg(args.cfg_file, args.set_cfgs)

    torch.cuda.set_device(get_local_rank())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    opt['_world_size'] = int(os.environ.get('WORLD_SIZE', 1))
    opt['_distributed'] = opt['_world_size'] > 1
    if opt['_distributed']:
        dist.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=opt['_world_size'],
            timeout=timedelta(minutes=20)
        )

    barrier() 

    root = opt.aux.logdir
    if get_local_rank() == 0:
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(root, 'states'), exist_ok=True)

    wandb_run = None
    if get_rank() == 0 and opt.aux.get('wandb_enable', False):
        wandb_run = create_wandb(opt)

    print0("==================")
    print0(opt.dump())
    print0("==================")
    print0(opt.aux.logdir)

    if get_local_rank() == 0:
        with open(os.path.join(root, 'opt.yaml'), 'w') as fd:
            fd.write(opt.dump())

    opt['_root'] = root
    opt['_resume'] = (
        os.path.exists(os.path.join(root, 'models', 'last.pth'))
        and os.path.exists(os.path.join(root, 'states', 'last.pth'))
    )


    if opt.task == 'mad':
        from libs.mad.worker import Trainer

    elif opt.task == 'svg': # Short Video Grounding: charades and TACoS
        from libs.svg.worker import Trainer

    elif opt.task == 'ego4d':         
        from libs.ego4d.worker import Trainer

    trainer = Trainer(opt, wandb_run=wandb_run)

    trainer.run()
    if opt['_distributed']:
        dist.destroy_process_group()

    if get_rank() == 0 and wandb_run is not None:
        wandb_run.finish()

    if get_rank() == 0:
        open(os.path.join(root, 'finished'), 'w').close()

if __name__ == '__main__':
    os.environ['DECORD_EOF_RETRY_MAX']='20480'


    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()


    main(int(os.environ.get('RANK', 0)), args)
