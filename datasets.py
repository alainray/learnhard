from typing import Any, Tuple
import torch.utils.data as nn
from os.path import join
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

def make_path(path):
    folder = path.split("_")[0]
    return join(folder, path)

class ImagenetCScore(nn.Dataset):
    def __init__(self, root: str="c_score/imagenet", 
                       train = True,
                       img_root=".",
                       transform=None,
                        **kwargs: Any):
        super(ImagenetCScore, self).__init__()
        # Load file list
        # Load scores list
        split = "train" if train else "test"
        self.transform = transform
        self.files = np.load(join(root,f"filenames_{split}.npy"), allow_pickle=True)
        self.img_root = img_root
        
        for i in range(len(self.files)):
            self.files[i] = make_path(str(self.files[i]).replace("b'","")[:-1])
        self.scores = np.load(join(root,f"scores_{split}.npy"))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.files[index]
        img = Image.open(join(self.img_root,img)) # open img in img folder
        if img.mode != "RGB":
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        if self.transform is not None:
            img = self.transform(img)

        score = self.scores[index]
        return index, img, score

# CIFAR10

def CIFARIdx(cl):
    dataset = "cifar10" if CIFAR10 else "cifar100"
    scores = np.load(f"c_score/{dataset}/scores.npy")

    class DatasetCIFARIdx(cl):
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return index, img, scores[index]

    return DatasetCIFARIdx