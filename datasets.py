from typing import Any, Tuple
import torch.utils.data as nn
from os.path import join
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from numpy import digitize, histogram_bin_edges

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
def get_bins(dataset, bin_type, n_bins):
    scores = np.load(f"c_score/{dataset}/scores.npy")
    if bin_type == "constant":
        bins = np.linspace(0,1,n_bins+1)
    else:
        bins = histogram_bin_edges(scores, n_bins)
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

def CIFARIdx(cl, label_type="score", bin_type="constant", n_bins=10):

    dataset = "cifar10" if cl == CIFAR10 else "cifar100"
    bins = get_bins(dataset,bin_type=bin_type,n_bins=n_bins)
    scores = np.load(f"c_score/{dataset}/scores.npy")
    class DatasetCIFARIdx(cl):
        
        def make_split(self, split):
            indices = np.load(f"c_score/{dataset}/indices_{split}.npy")
            self.data = [self.data[index] for index in indices]
            #self.data = self.data[indices]
            #print(indices)
            self.targets = [self.targets[index] for index in indices]
            # self.targets = self.targets[indices]
            nonlocal scores
            scores = [scores[index] for index in indices]
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            label = scores[index] if label_type=="score" else (digitize(scores[index],bins) - 1).astype(np.longlong)
            return index, img, label

    return DatasetCIFARIdx


if __name__ == "__main__":
    dataset = "cifar100"
    split = "train" 
    indices = np.load(f"c_score/{dataset}/indices_{split}.npy")
    print(indices)