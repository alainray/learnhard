from typing import Any, Tuple
import torch.utils.data as nn
from os.path import join
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from numpy import digitize, histogram_bin_edges
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(args):
    if args.dataset == "imagenet":
        root = "/workspace1/araymond/ILSVRC2012/train/"
    else:
        root = "."
    data = get_datasets(args)
    train_data = data[dataset](transform=preproc['train'][args.res], root=root, train=True, download=True)
    test_data = data[dataset](transform=preproc['test'][args.res], 
                            root=root, 
                            train=False if "cifar" not in dataset else True, 
                            download=True)
    if "cifar" in dataset:
        train_data.make_split("train")
        test_data.make_split("test")
    if args.test_ds != "":
        test_data2 = data[args.test_ds](transform=preproc['test'][args.res],
                                        root=root, 
                                        train=False if "cifar" not in args.test_ds else True, 
                                        download=True)
        if "cifar" in args.test_ds:
            test_data2.make_split("all")
    
        test_dl2 = DataLoader(test_data2, batch_size=args.test_bs)
    
    train_dl = DataLoader(train_data, batch_size=args.train_bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=args.test_bs)
    
    return train_dl, [test_dl, test_dl2]

def CIFARIdx(cl, args):

    dataset = "cifar10" if cl == CIFAR10 else "cifar100"
    bins = get_bins(args)
    scores = np.load(f"c_score/{dataset}/scores.npy")
    class DatasetCIFARIdx(cl):
        
        def make_split(self, split):
            if split != "all":
                indices = np.load(f"c_score/{dataset}/indices_{split}.npy")
                self.data = [self.data[index] for index in indices]
                #self.data = self.data[indices]
                #print(indices)
                self.targets = [self.targets[index] for index in indices]
                # self.targets = self.targets[indices]
                nonlocal scores
                self.scores = [scores[index] for index in indices]
            else:
                self.scores = scores
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            label = self.scores[index] if label_type=="score" else (digitize(self.scores[index],bins) - 1).astype(np.longlong)
            return index, img, label

    return DatasetCIFARIdx


def make_path(path):
    folder = path.split("_")[0]
    return join(folder, path)


class ImagenetCScore(nn.Dataset):
    def __init__(self, root: str=".", 
                       train = True,
                       img_root="c_score/imagenet",
                       transform=None,
                        **kwargs: Any):
        super(ImagenetCScore, self).__init__()
        # Load file list
        # Load scores list
        split = "train" if train else "test"
        self.transform = transform
        self.files = np.load(join(root,f"filenames_{split}.npy"), allow_pickle=True)
        self.root = root
        
        for i in range(len(self.files)):
            self.files[i] = make_path(str(self.files[i]).replace("b'","")[:-1])
        self.scores = np.load(join(img_root,f"scores_{split}.npy"))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.files[index]
        img = Image.open(join(self.root,img)) # open img in img folder
        if img.mode != "RGB":
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        if self.transform is not None:
            img = self.transform(img)

        score = self.scores[index]
        return index, img, score

# CIFAR10
def get_bins(args):
    scores = np.load(f"c_score/{args.dataset}/scores.npy")
    if args.bin_type == "constant":
        bins = np.linspace(0,1,args.n_bins+1)
    else:
        bins = histogram_bin_edges(scores, args.n_bins)
    delta = 0.00001
    bins[0] -= delta
    bins[-1] += delta
    return bins

def get_class_weights(dataset, bins):
    scores = np.load(f"c_score/{dataset}/scores.npy")
    len_dataset = scores.shape[0]
    n_bins = bins.shape[0]-1
    weights = [0.0 for i in range(n_bins)]
    for index in range(len_dataset):
        weights[digitize(scores[index],bins)-1]+=1
    print(weights)
    weights = np.array(weights)/len_dataset
    weights = np.reciprocal(weights)
    weights[weights==np.inf] = 0.0
    max_weight = weights[weights>0.0].max()

    return weights/max_weight

'''
Transforms and Data Parameters
'''
def get_datasets(args):

    data = {"imagenet": ImagenetCScore,
            "cifar10": CIFARIdx(CIFAR10,args),
            "cifar100": CIFARIdx(CIFAR100, args)}
    return data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preproc = { 'train':{224: transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]),
                            32: transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
                        },
            'test':{224: transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]),
                    32: transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize,
                                        ])           
                    }
}


if __name__ == "__main__":
    dataset = "cifar100"
    split = "train" 
    indices = np.load(f"c_score/{dataset}/indices_{split}.npy")
    print(indices)