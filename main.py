import comet_ml
import torch
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from datasets import ImagenetCScore, CIFARIdx, get_class_weights
from utils import AverageMeter, checkpoint
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import  MSELoss, CrossEntropyLoss
from utils import train, test, setup_comet
import numpy as np
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("arch")                     # Architecture: fc, minialex
parser.add_argument("lr", type=float)
parser.add_argument("--dataset", default="imagenet") 
parser.add_argument("--seed", type=int, default=123)                     # Random seed (an int)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--label_type", type=str, default="score") # score/bins
parser.add_argument("--bin_type", type=str, default="equal") # constant/equal
parser.add_argument("--n_bins", type=int, default=10)
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--cometKey", type=str)
parser.add_argument("--cometWs", type=str)
parser.add_argument("--cometName", type=str)
args = parser.parse_args()
seed = args.seed
seed = args.seed
arch = args.arch
lr = args.lr
n_epochs = args.epochs
label_type = args.label_type
torch.manual_seed(seed)

# dataset "MNIST", "CIFAR10"

dataset = args.dataset
if dataset == "imagenet":
    root = "/workspace1/araymond/ILSVRC2012/train/"
else:
    root = "."

data_params = {'label_type': label_type, 'n_bins': args.n_bins, 'bin_type': args.bin_type}
data = {"imagenet": ImagenetCScore,
         "cifar10": CIFARIdx(CIFAR10,**data_params),
        "cifar100": CIFARIdx(CIFAR100, **data_params)}
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
n_classes = {'imagenet': 10, 'cifar10': 10, 'cifar100': 10} # regression task
models = {'resnet18': resnet18, "resnet34": resnet34, "resnet50": resnet50} 
optimizers = {'adam': Adam, 'sgd': SGD}
input_dims = {'imagenet': 224*224*3}
optimizer = args.opt
device = 'cuda'
criterion = MSELoss() if label_type == "score" else CrossEntropyLoss(weight=get_class_weights(dataset,args.n_bins))

pretrained = True
freeze = False
model = models[arch](pretrained=pretrained)

if freeze:
    # Freeze layers!
    for param in model.parameters():
        param.requires_grad = False

# change last layer for regression between 0 and 1
# model.fc = nn.Sequential(nn.Linear(in_features[arch],1), nn.Sigmoid())
if label_type == "score":
    model.fc = nn.Linear(in_features[arch],1)
else:
    model.fc = nn.Linear(in_features[arch],args.n_bins)

opt_params = {"lr": lr}

if optimizer=="sgd":
    opt_params['momentum'] = 0.9 

opt = optimizers[optimizer](filter(lambda p: p.requires_grad, model.parameters()), **opt_params)

train_dl = DataLoader(train_data, batch_size=256, shuffle=True)
test_dl = DataLoader(test_data, batch_size=512)


model.to(device)

exp = setup_comet(args)
exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
model.comet_experiment_key = exp.get_key() # To retrieve existing experiment

for epoch in range(1, n_epochs + 1):
    model, stats = train(exp, args, model, train_dl, opt, device, criterion, epoch)
    checkpoint(args, model, stats, epoch, split="train")
    # print(f"\nTest Epoch {epoch}", flush=True)
    # model, stats = test(args, model, test_dl, device, criterion)
    #checkpoint(args, model, stats, epoch, split="test")