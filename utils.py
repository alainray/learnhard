from comet_ml import Experiment, ExistingExperiment
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import os
from os import mkdir
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from functools import wraps
from time import time
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("arch")            # resnet18,34,50/simplecnn
    parser.add_argument("lr", type=float)
    parser.add_argument("--dataset", type=str, default="imagenet") # imagenet/cifar10/cifar100
    parser.add_argument("--test_ds", type=str, default="") 
    parser.add_argument("--seed", type=int, default=123)        # Random seed (an int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warm", type=int, default=1)
    parser.add_argument("--res", type=int, default=224) # Input Img Resolution 224/32
    parser.add_argument("--label_type", type=str, default="score")  # score/bins
    parser.add_argument("--bin_type", type=str, default="equal")    # constant/equal
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--class_weighting", type=str, default="y")
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--cometKey", type=str)
    parser.add_argument("--cometWs", type=str)
    parser.add_argument("--cometName", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_bs", type=int, default=256)
    parser.add_argument("--test_bs", type=int, default=512)
    parser.add_argument("--loss", type=str, default="")
    parser.add_argument("--decay", type=float, default=0.0)
    return parser.parse_args()

def set_random_state(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)                          

# Comet Experiments
def setup_comet(args, resume_experiment_key=''):
    api_key = args.cometKey # w4JbvdIWlas52xdwict9MwmyH
    workspace = args.cometWs # alainray
    project_name = args.cometName # learnhard
    enabled = bool(api_key) and bool(workspace)
    disabled = not enabled

    print(f"Setting up comet logging using: {{api_key={api_key}, workspace={workspace}, enabled={enabled}}}")

    if resume_experiment_key:
        experiment = ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key)
        return experiment

    experiment = Experiment(api_key=api_key, parse_args=False, project_name=project_name,
                            workspace=workspace, disabled=disabled)
    # TEST
    experiment_name = get_prefix(args)
    if experiment_name:
        experiment.set_name(experiment_name)

    train_data_type = os.environ.get('TRAIN_DATA_TYPE')
    if train_data_type:
        experiment.add_tag(train_data_type)

    tags = os.environ.get('TAGS')
    if tags:
        experiment.add_tags(tags.split(','))

    return experiment


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Func: {f.__name__} took: {te-ts:0.0f} sec")
        return result
    return wrap



def create_splits(f, s, l, root="c_score/imagenet"):
    d= {'train': {'filenames': None, 'scores': None, 'labels': None},
            'test': {'filenames': None, 'scores': None, 'labels': None}}
    
    f_tr, f_ts, s_tr, s_ts, l_tr, l_ts = train_test_split(f,s,l, test_size=0.2, random_state=123)
    
    d['train']['filenames'] = f_tr
    d['train']['scores'] = s_tr
    d['train']['labels'] = l_tr
    d['test']['filenames'] = f_ts
    d['test']['scores'] = s_ts
    d['test']['labels'] = l_ts

    print("Armando archivos...")
    for split, v in d.items():
        for name in v.keys():
            np.save(join(root,f"{name}_{split}.npy"), d[split][name])

def get_prefix(args):
    return "_".join([str(w) for k,w in vars(args).items() if "comet" not in k])
def checkpoint(args, model, stats, epoch, root="ckpts", res_root="stats", split="train"):
    
    prefix = get_prefix(args)
    
    if split == "train":
        if not os.path.isdir(root):
            mkdir(root)

        torch.save(model.state_dict(), f"{root}/{prefix}_{split}_{epoch}.pth")
        
    if not os.path.isdir(res_root):
        mkdir(res_root)
    torch.save(stats, f"{res_root}/{prefix}_{split}_{epoch}.pth")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    root = "c_score/cifar10"
    #filenames = np.load(join(root,"scores.npy"), allow_pickle=True)
    filenames = np.array(list(range(50000)))
    scores = np.load(join(root,"scores.npy"), allow_pickle=True)
    labels = np.load(join(root,"labels.npy"), allow_pickle=True)
    create_splits(filenames, scores, labels, root=root)