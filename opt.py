import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import _LRScheduler
from datasets import get_bins, get_class_weights
from train import BPRLoss
def get_opt(args, model):
    optimizers = {'adam': Adam, 'sgd': SGD}
    optimizer = args.opt

    opt_params = {"lr": args.lr, 'weight_decay': args.decay}

    if optimizer=="sgd":
        opt_params['momentum'] = 0.9

    opt = optimizers[optimizer](filter(lambda p: p.requires_grad, model.parameters()), **opt_params)
    return opt

def get_criterion(args):
    bins = get_bins(args)
    if args.class_weighting == "y":
        class_weights = torch.from_numpy(get_class_weights(args.dataset,bins)).float().cuda()
    else:
        class_weights = torch.ones(args.n_bins).float().cuda()

    if args.label_type == "score" and args.loss != "bpr":
        return MSELoss()
    elif args.label_type == "score" and args.loss == "bpr":
        return BPRLoss
    else:
        return CrossEntropyLoss(weight=class_weights)

def get_scheduler(scheduler_name, optimizer, num_epochs, **kwargs):
  if scheduler_name == 'constant':
    return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)

  elif scheduler_name == 'step2':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 2), gamma=0.1, **kwargs)
  elif scheduler_name == 'step3':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
  elif scheduler_name == 'exponential':
    return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
  elif scheduler_name == 'cosine':
    return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
  elif scheduler_name == 'step-more':
    return lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, **kwargs)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]