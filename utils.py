import numpy as np
import torch
import os
from os import mkdir
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from functools import wraps
from time import time

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

def checkpoint(model, stats, epoch, arch, dataset, root="ckpts", res_root="stats", split="train"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(model.state_dict, f"{root}/{arch}_{dataset}_{split}_{epoch}.pth")
    if not os.path.isdir(res_root):
        mkdir(res_root)
    torch.save(stats, f"{res_root}/{arch}_{dataset}_{split}_{epoch}.pth")

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
    root = "c_score/imagenet"
    filenames = np.load(join(root,"filenames.npy"), allow_pickle=True)
    scores = np.load(join(root,"scores.npy"), allow_pickle=True)
    labels = np.load(join(root,"labels.npy"), allow_pickle=True)
    create_splits(filenames, scores, labels)