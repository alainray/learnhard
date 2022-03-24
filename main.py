import torch
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from datasets import ImagenetCScore, CIFARIdx
from utils import AverageMeter, checkpoint
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import  MSELoss
from utils import train, test
import numpy as np
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("arch")                     # Architecture: fc, minialex
parser.add_argument("lr", type=float)
parser.add_argument("--dataset", default="imagenet") 
parser.add_argument("--seed", type=int, default=123)                     # Random seed (an int)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()
seed = args.seed
seed = args.seed
arch = args.arch
lr = args.lr
n_epochs = args.epochs
torch.manual_seed(seed)

# dataset "MNIST", "CIFAR10"

dataset = args.dataset
if dataset == "imagenet":
    root = "/workspace1/araymond/ILSVRC2012/train/"
else:
    root = "."
data = {'imagenet': ImagenetCScore, "cifar10": CIFARIdx(CIFAR10), "cifar100": CIFARIdx(CIFAR100)}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocessing_tr = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

preprocessing_ts = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
train_data = data[dataset](transform=preprocessing_tr, root=root, train=True, download=True)
test_data = data[dataset](transform=preprocessing_ts, root=root, train=False, download=True)

in_features = {'resnet18': 512, "resnet34": 512, "resnet50": 2048} 
in_channels = {'imagenet': 3, "cifar10": 3, "cifar100": 3}
n_classes = {'imagenet': 1, 'cifar10': 1, 'cifar100': 1} # regression task
models = {'resnet18': resnet18, "resnet34": resnet34, "resnet50": resnet50} 
optimizers = {'adam': Adam, 'sgd': SGD}
input_dims = {'imagenet': 224*224*3}
optimizer = "sgd"  # "sgd", "adam"
device = 'cuda'
criterion = MSELoss()


pretrained = True
model = models[arch](pretrained=pretrained)

if pretrained:
    # Freeze layers!
    for param in model.parameters():
        param.requires_grad = False

# change last layer for regression between 0 and 1
# model.fc = nn.Sequential(nn.Linear(in_features[arch],1), nn.Sigmoid())
model.fc = nn.Linear(in_features[arch],1)
model.fc.bias.requires_grad = False
model.fc.bias.fill_(0.5)
model.fc.bias.requires_grad = True

n_gpus = 1
if n_gpus > 1:
    model = nn.DataParallel(model, list(range(n_gpus)))

opt = optimizers[optimizer](filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

train_dl = DataLoader(train_data, batch_size=256)
test_dl = DataLoader(test_data, batch_size=512)

if n_gpus > 1:
    model.to(0)
else:    
    model.to(device)
for epoch in range(1, n_epochs + 1):
    print(f"\nTrain Epoch {epoch}", flush=True)
    model, stats = train(model, train_dl, opt, device, criterion)
    checkpoint(args, model, stats, epoch, split="train")
    print(f"\nTest Epoch {epoch}", flush=True)
    model, stats = test(model, test_dl, device, criterion)
    checkpoint(args, model, stats, epoch, split="test")