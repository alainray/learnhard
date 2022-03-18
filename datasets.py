from typing import Any
import torch.utils.data as nn
from os.path import join
from PIL import Image
import numpy as np

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

        if self.transform is not None:
            img = self.transform(img)

        score = self.scores[index]
        return index, img, score
