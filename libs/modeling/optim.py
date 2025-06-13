from bisect import bisect_right
from collections import Counter
import math
import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from .blocks import MaskedConv1D, LayerNorm, Scale, LayerScale

# snag_whitelist_modules = (nn.Linear, nn.Conv1d, MaskedConv1D)
# snag_blacklist_modules = (LayerNorm, Scale, LayerScale)
# def snag_filter(mn, m, pn, p, decay, no_decay):
#     fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#     captured = True
#     if pn.endswith('bias'):
#         no_decay.add(fpn)
#     elif pn.endswith('weight') and isinstance(m, snag_whitelist_modules):
#         decay.add(fpn)
#     elif pn.endswith('weight') and isinstance(m, snag_blacklist_modules):
#         no_decay.add(fpn)
#     elif pn.endswith('scale') and isinstance(m, snag_blacklist_modules):
#         no_decay.add(fpn)
#     elif pn.endswith('bkgd_token'):
#         no_decay.add(fpn)
#     else:
#         captured = False
    
#     return captured

# cocap_whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
# cocap_blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)
# def cocap_filter(mn, m, pn, p, decay, no_decay):
#     fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

#     captured = True
#     if pn.endswith("bias"):
#         no_decay.add(fpn)
#     elif pn.endswith("proj") or pn.endswith("projection"):
#         decay.add(fpn)
#     elif fpn.endswith("embedding"):
#         no_decay.add(fpn)
#     elif pn.endswith("weight") and isinstance(m, cocap_whitelist_weight_modules):
#         decay.add(fpn)
#     elif pn.endswith("weight") and isinstance(m, cocap_blacklist_weight_modules):
#         no_decay.add(fpn)
#     else:
#         captured = False
    
#     return captured

# def additional_infuse(mn, m, pn, p, decay, no_decay):
#     fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

#     captured = False
#     if fpn.endswith("pos_embedding"):
#         no_decay.add(fpn)
#         captured = True
#     else:
#         captured = cocap_filter(mn, m, pn, p, decay, no_decay) # HACK
    
#     return captured

def make_optimizer(model, opt):
    decay, no_decay = set(), set()
    whitelist_modules = (nn.Linear, nn.Conv1d, MaskedConv1D, nn.Conv2d)
    blacklist_modules = (LayerNorm, Scale, LayerScale, nn.LayerNorm, nn.Embedding, nn.BatchNorm2d, nn.BatchNorm3d)

    # has_mv_enc = (model.mv_enc is not None)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if '.' in pn:
                continue 
            
            if not p.requires_grad:
                continue
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_modules):
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, blacklist_modules):
                no_decay.add(fpn)
            elif pn.endswith('bkgd_token'):
                no_decay.add(fpn)
            # elif fpn.startswith("mv_enc") and cocap_filter(mn, m, pn, p, decay, no_decay):
            #     pass
            # elif fpn.startswith("input_fusion") and additional_infuse(mn, m, pn, p, decay, no_decay):
            #     pass
            # elif pn == 'in_proj_weight' and ('attn' in mn):
            #     no_decay.add(fpn)
            # elif pn.endswith('embedding') and ('mv_enc' in mn):
            #     no_decay.add(fpn)
            # elif fpn.endswith('spatial_enc.proj'):
            #     decay.add(fpn)
            # elif fpn.endswith('pe'):
            #     decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, (
        'parameters {:s} made it into both decay/no decay sets'
        ''.format(str(inter_params))
    )
    assert len(param_dict.keys() - union_params) == 0, (
        'parameters {:s} were not separated into either decay/no decay set'
        ''.format(str(param_dict.keys() - union_params))
    )

    decay = sorted(list(decay))
    no_decay = sorted(list(no_decay))

    param_groups = [
        {'params': [param_dict[pn] for pn in decay],
         'weight_decay': opt['weight_decay'],
         'lr': opt['lr'],
        },
        {'params': [param_dict[pn] for pn in no_decay],
         'weight_decay': 0.0,
         'lr': opt['lr'],
        },
    ]

    # vid_groups = [
    #     {'params': [param_dict[pn] \
    #         for pn in decay if 'vid_net' in pn],
    #      'weight_decay': opt['weight_decay'],
    #      'lr': opt.get('vid_net_lr', opt['lr']),
    #     },
    #     {'params': [param_dict[pn] \
    #         for pn in no_decay if 'vid_net' in pn],
    #      'weight_decay': 0.0,
    #      'lr': opt.get('vid_net_lr', opt['lr']),
    #     },
    # ]

    # text_groups = [
    #     {'params': [param_dict[pn] \
    #         for pn in decay if 'text_net' in pn],
    #      'weight_decay': opt['weight_decay'],
    #      'lr': opt.get('text_net_lr', opt['lr']),
    #     },
    #     {'params': [param_dict[pn] \
    #         for pn in no_decay if 'text_net' in pn],
    #      'weight_decay': 0.0,
    #      'lr': opt.get('text_net_lr', opt['lr']),
    #     },
    # ]

    # fusion_groups = [
    #     {'params': [param_dict[pn] \
    #         for pn in decay if 'fusion' in pn],
    #      'weight_decay': opt['weight_decay'],
    #      'lr': opt.get('fusion_lr', opt['lr']),
    #     },
    #     {'params': [param_dict[pn] \
    #         for pn in no_decay if 'fusion' in pn],
    #      'weight_decay': 0.0,
    #      'lr': opt.get('fusion_lr', opt['lr']),
    #     },
    # ]

    # cls_head_groups = [
    #     {'params': [param_dict[pn] \
    #         for pn in decay if 'cls_head' in pn],
    #      'weight_decay': opt['weight_decay'],
    #      'lr': opt.get('cls_head_lr', opt['lr']),
    #     },
    #     {'params': [param_dict[pn] \
    #         for pn in no_decay if 'cls_head' in pn],
    #      'weight_decay': 0.0,
    #      'lr': opt.get('cls_head_lr', opt['lr']),
    #     },
    # ]

    # reg_head_groups = [
    #     {'params': [param_dict[pn] \
    #         for pn in decay if 'reg_head' in pn],
    #      'weight_decay': opt['weight_decay'],
    #      'lr': opt.get('reg_head_lr', opt['lr']),
    #     },
    #     {'params': [param_dict[pn] \
    #         for pn in no_decay if 'reg_head' in pn],
    #      'weight_decay': 0.0,
    #      'lr': opt.get('reg_head_lr', opt['lr']),
    #     },
    # ]

    # param_groups = (
    #     vid_groups + text_groups + fusion_groups
    #     + cls_head_groups + reg_head_groups
    # )

    # # if opt.model.mv_enc.name is not None:
    # if model.mv_enc is not None or True:
    #     mv_enc_groups = [
    #         {'params': [param_dict[pn] \
    #             for pn in decay if 'mv_enc' in pn],
    #         'weight_decay': opt['weight_decay'],
    #         'lr': opt.get('reg_head_lr', opt['lr']),
    #         },
    #         {'params': [param_dict[pn] \
    #             for pn in no_decay if 'mv_enc' in pn],
    #         'weight_decay': 0.0,
    #         'lr': opt.get('reg_head_lr', opt['lr']),
    #         },
    #     ]
    #     param_groups.extend(mv_enc_groups)

    if opt['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=opt['lr'],
            momentum=opt.get('momentum', 0.9),
            weight_decay=opt.get('weight_decay', 0)
        )
    elif opt['name'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt['lr'], betas=(0.9, 0.999),
            weight_decay=opt.get('weight_decay', 0)
        )
    elif opt['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt['lr'], betas=(0.9, 0.999),
            weight_decay=opt.get('weight_decay', 0)
        )
    else:
        raise NotImplementedError(f"invalid optimizer: {opt['name']}")

    return optimizer

# def make_optimizer_pretrain(model, opt):
#     decay, no_decay = set(), set()

#     for mn, m in model.named_modules():
#         for pn, p in m.named_parameters():
#             if '.' in pn:
#                 continue 
            
#             if not p.requires_grad:
#                 continue

#             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#             if cocap_filter(mn, m, pn, p, decay, no_decay):
#                 pass
#             elif fpn.endswith('pe'):
#                 print(fpn)
#                 no_decay.add(fpn)
#             elif fpn.endswith('mask_out_token'):
#                 print(fpn)
#                 no_decay.add(fpn)

#     param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
#     inter_params = decay & no_decay
#     union_params = decay | no_decay
#     assert len(inter_params) == 0, (
#         'parameters {:s} made it into both decay/no decay sets'
#         ''.format(str(inter_params))
#     )
#     assert len(param_dict.keys() - union_params) == 0, (
#         'parameters {:s} were not separated into either decay/no decay set'
#         ''.format(str(param_dict.keys() - union_params))
#     )

#     decay = sorted(list(decay))
#     no_decay = sorted(list(no_decay))

#     param_groups = [
#         {'params': [param_dict[pn] for pn in decay],
#          'weight_decay': opt['weight_decay'],
#          'lr': opt['lr'],
#         },
#         {'params': [param_dict[pn] for pn in no_decay],
#          'weight_decay': 0.0,
#          'lr': opt['lr'],
#         },
#     ]

#     if opt['name'] == 'sgd':
#         optimizer = torch.optim.SGD(
#             param_groups,
#             lr=opt['lr'],
#             momentum=opt.get('momentum', 0.9),
#             weight_decay=opt.get('weight_decay', 0)
#         )
#     elif opt['name'] == 'adam':
#         optimizer = torch.optim.Adam(
#             param_groups,
#             lr=opt['lr'], betas=(0.9, 0.999),
#             weight_decay=opt.get('weight_decay', 0)
#         )
#     elif opt['name'] == 'adamw':
#         optimizer = torch.optim.AdamW(
#             param_groups,
#             lr=opt['lr'], betas=(0.9, 0.999),
#             weight_decay=opt.get('weight_decay', 0)
#         )
#     else:
#         raise NotImplementedError(f"invalid optimizer: {opt['name']}")

#     return optimizer


# def make_timesformer_optimizer(model, opt):
#     """
#     Construct a stochastic gradient descent or ADAM optimizer with momentum.
#     Details can be found in:
#     Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
#     and
#     Diederik P.Kingma, and Jimmy Ba.
#     "Adam: A Method for Stochastic Optimization."

#     Args:
#         model (model): model to perform stochastic gradient descent
#         optimization or ADAM optimization.
#         cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
#         learning rate,  momentum, weight_decay, dampening, and etc.
#     """
#     # Batchnorm parameters.
#     bn_params = []
#     # Non-batchnorm parameters.
#     non_bn_parameters = []
#     for name, p in model.named_parameters():
#         if "bn" in name:
#             bn_params.append(p)
#         else:
#             non_bn_parameters.append(p)
#     # Apply different weight decay to Batchnorm and non-batchnorm parameters.
#     # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
#     # Having a different weight decay on batchnorm might cause a performance
#     # drop.
#     BN_WEIGHT_DECAY = 0
#     SOLVER_WEIGHT_DECAY = opt.weight_decay
#     optim_params = [
#         {"params": bn_params, "weight_decay": BN_WEIGHT_DECAY},
#         {"params": non_bn_parameters, "weight_decay": SOLVER_WEIGHT_DECAY},
#     ]
#     # Check all parameters will be passed into optimizer.
#     assert len(list(model.parameters())) == len(non_bn_parameters) + len(
#         bn_params
#     ), "parameter size does not match: {} + {} != {}".format(
#         len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
#     )

#     if opt.name == "sgd":
#         return torch.optim.SGD(
#             optim_params,
#             lr=opt.lr,
#             momentum=opt.momentum,
#             weight_decay=opt.weight_decay,
#             dampening=opt.dampening,
#             nesterov=opt.netserov,
#         )
#     elif opt.name == "adam":
#         return torch.optim.Adam(
#             optim_params,
#             lr=opt.lr,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=opt.weight_decay,
#         )
#     elif opt.name == "adamw":
#         return torch.optim.AdamW(
#             optim_params,
#             lr=opt.lr,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=opt.weight_decay,
#         )
#     else:
#         raise NotImplementedError(
#             "Does not support {} optimizer".format(opt.name)
#         )

# def make_timesformer_optimizer_v2(model, opt):
#     """
#     Construct a stochastic gradient descent or ADAM optimizer with momentum.
#     Details can be found in:
#     Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
#     and
#     Diederik P.Kingma, and Jimmy Ba.
#     "Adam: A Method for Stochastic Optimization."

#     Args:
#         model (model): model to perform stochastic gradient descent
#         optimization or ADAM optimization.
#         cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
#         learning rate,  momentum, weight_decay, dampening, and etc.
#     """
#     # Batchnorm parameters.
#     bn_params = []
#     # Non-batchnorm parameters.
#     non_bn_parameters = []
#     for name, p in model.named_parameters():
#         if "bn" in name:
#             bn_params.append(p)
#         if ('pos_embed') in name or ('cls_token') in name:
#             bn_params.append(p)
#         else:
#             non_bn_parameters.append(p)
#     # Apply different weight decay to Batchnorm and non-batchnorm parameters.
#     # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
#     # Having a different weight decay on batchnorm might cause a performance
#     # drop.
#     BN_WEIGHT_DECAY = 0
#     SOLVER_WEIGHT_DECAY = opt.weight_decay
#     optim_params = [
#         {"params": bn_params, "weight_decay": BN_WEIGHT_DECAY},
#         {"params": non_bn_parameters, "weight_decay": SOLVER_WEIGHT_DECAY},
#     ]
#     # Check all parameters will be passed into optimizer.
#     assert len(list(model.parameters())) == len(non_bn_parameters) + len(
#         bn_params
#     ), "parameter size does not match: {} + {} != {}".format(
#         len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
#     )

#     if opt.name == "sgd":
#         return torch.optim.SGD(
#             optim_params,
#             lr=opt.lr,
#             momentum=opt.momentum,
#             weight_decay=opt.weight_decay,
#             dampening=opt.dampening,
#             nesterov=opt.netserov,
#         )
#     elif opt.name == "adam":
#         return torch.optim.Adam(
#             optim_params,
#             lr=opt.lr,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=opt.weight_decay,
#         )
#     elif opt.name == "adamw":
#         return torch.optim.AdamW(
#             optim_params,
#             lr=opt.lr,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=opt.weight_decay,
#         )
#     else:
#         raise NotImplementedError(
#             "Does not support {} optimizer".format(opt.name)
#         )

def make_optimizer_simple(model, opt):
    param_groups = model.parameters()
    if opt['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=opt['lr'],
            momentum=opt.get('momentum', 0.9),
            weight_decay=opt.get('weight_decay', 0)
        )
    elif opt['name'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt['lr'], betas=(0.9, 0.999),
            weight_decay=opt.get('weight_decay', 0)
        )
    elif opt['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt['lr'], betas=(0.9, 0.999),
            weight_decay=opt.get('weight_decay', 0)
        )
    else:
        raise NotImplementedError(f"invalid optimizer: {opt['name']}")

    return optimizer


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=1e-8,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class LinearWarmupMultiStepLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a multi-step schedule that decays
    the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        milestones,
        warmup_start_lr=0.0,
        gamma=0.1,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            milestones (list): List of epoch indices. Must be increasing.
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.milestones = Counter(milestones)
        self.gamma = gamma

        super(LinearWarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - self.warmup_epochs) not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch - self.warmup_epochs]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch - self.warmup_epochs)
                for base_lr in self.base_lrs]


def make_scheduler(optimizer, opt):
    warmup_itrs = opt.get('warmup_epochs', 0) * opt['itrs_per_epoch']

    if opt['name'] == 'cosine':
        max_itrs = warmup_itrs + opt['epochs'] * opt['itrs_per_epoch']
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_itrs,
            max_epochs=max_itrs
        )
    elif opt['name'] == 'multistep':
        step_itrs = [opt['itrs_per_epoch'] * s for s in opt['steps']]
        scheduler = LinearWarmupMultiStepLR(
            optimizer,
            warmup_epochs=warmup_itrs,
            milestones=step_itrs,
            gamma=opt.get('gamma', 0.1)
        )
    elif opt['name'] == 'null':
        # step_itrs = [opt['itrs_per_epoch'] * s for s in opt['steps']]
        # scheduler = LinearWarmupMultiStepLR(
        #     optimizer,
        #     warmup_epochs=warmup_itrs,
        #     milestones=step_itrs,
        #     gamma=opt.get('gamma', 0.1)
        # )
        scheduler = None
    else:
        raise NotImplementedError(f"invalid scheduler: {opt['name']}")

    return scheduler
