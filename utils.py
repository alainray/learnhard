import numpy as np
import torch
import os
from os import mkdir
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split

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
    

def write_ordering(n, stats, iteration, arch, dataset, root="results", split="train"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(stats, f"{root}/{arch}_{dataset}_{split}_{iteration}_{n}_order.pth")


def checkpoint(model, stats, epoch, arch, dataset, root="results", split="train"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(model.state_dict, f"{root}/{arch}_{dataset}_{split}_{epoch}.pth")


def record_stats(stats, iteration, arch, dataset, seed, optimizer, root="results", split="train", train_mode="regular"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(stats, f"{root}/{arch}_{dataset}_{train_mode}_{optimizer}_{split}_{iteration}_{seed}_stats.pth")


def record_affected(stats, iteration, arch, dataset, seed, optimizer, root="results", split="train", train_mode="regular"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(stats, f"{root}/{arch}_{dataset}_{train_mode}_{optimizer}_{split}_{iteration}_{seed}_affected.pth")


def record_metrics(stats, iteration, arch, dataset, seed, optimizer, root="results", split="train", train_mode="regular"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(stats, f"{root}/{arch}_{dataset}_{train_mode}_{optimizer}_{split}_{iteration}_{seed}_metrics.pth")


def get_layers(arch, dataset):
    # Let's determine which parameter groups are in each layer according to architecture
    if arch == 'resnet18':
        if dataset == "cifar100":
            layers = {"Conv1": [0, 580], "Block 1": [580, 15140],
                      "Block 2": [15140, 66740], "Block 3": [66740, 272340],
                      "Block 4": [272340, 1093140], "Classifier": [1093140, 1157140]
                      }
        elif dataset == "cifar10":
            layers = {"Conv1": [0, 580], "Block 1": [580, 15140],
                      "Block 2": [15140, 66380], "Block 3": [66380, 272340],
                      "Block 4": [272340, 1093140], "Classifier": [1093140, 1099540]
                      }

        elif dataset == "mnist":
            layers = {"Conv1": [0, 220], "Block 1": [220, 14780],
                      "Block 2": [14780, 66740], "Block 3": [66740, 271980],
                      "Block 4": [271980, 1092780], "Classifier": [1092780, 1099180]
                      }

        result = {"Conv1": 0, "Block 1": 0, "Block 2": 0, "Block 3": 0, "Block 4": 0, "Classifier": 0}

    elif arch == 'minialex':
        if dataset == "cifar100":
            layers = {"Layer 1": [0, 15200], "Layer 2": [15200, 1015400],
                      "Layer 3": [1015400, 3780584], "Layer 4": [3780584, 3854504],
                      "Classifier": [3854504, 3873804]
                      }
        elif dataset == "cifar10":
            layers = {"Layer 1": [0, 15200], "Layer 2": [15200, 1015400],
                      "Layer 3": [1015400, 3780584], "Layer 4": [3780584, 3854504],
                      "Classifier": [3854504, 3856434]
                      }

        elif dataset == "mnist":
            layers = {"Layer 1": [0, 5200], "Layer 2": [5200, 1005400],
                      "Layer 3": [1005400, 3770584], "Layer 4": [3770584, 3844504],
                      "Classifier": [3844504, 3846434]
                      }
        result = {"Layer 1": 0, "Layer 2": 0, "Layer 3": 0, "Layer 4": 0, "Classifier": 0}

    elif arch == 'fc':
        if dataset == "cifar100":
            layers = {"Layer 1": [0, 3002321], "Layer 2": [3002321, 3306479],
                      "Layer 3": [3306479, 3337367], "Layer 4": [3337367, 3340467],
                      "Classifier": [3340467, 3343667]
                      }
        elif dataset == "cifar10":
            layers = {"Layer 1": [0, 3002321], "Layer 2": [3002321, 3306479],
                      "Layer 3": [3306479, 3337367], "Layer 4": [3337367, 3340467],
                      "Classifier": [3340467, 3340787]
                      }

        elif dataset == "mnist":
            layers = {"Layer 1": [0, 766945], "Layer 2": [766945, 1071103],
                      "Layer 3": [1071103, 1101991], "Layer 4": [1101991, 1105091],
                      "Classifier": [1105091, 1105411]
                      }

        result = {"Layer 1": 0, "Layer 2": 0, "Layer 3": 0, "Layer 4": 0, "Classifier": 0}

    return layers, result



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