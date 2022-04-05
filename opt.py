import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from datasets import get_bins, get_class_weights
def get_opt(args, model):
    optimizers = {'adam': Adam, 'sgd': SGD}
    optimizer = args.opt

    opt_params = {"lr": args.lr}

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
    return MSELoss() if args.label_type == "score" else CrossEntropyLoss(weight=class_weights)
    