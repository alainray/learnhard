import torch
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.transforms as transforms
from datasets import ImagenetCScore
from utils import AverageMeter, checkpoint
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import  MSELoss
from utils import timing
import numpy as np
import torch.nn as nn
seed = 123
torch.manual_seed(seed)

# dataset "MNIST", "CIFAR10"

dataset = "imagenet"
img_root = "/mnt/nas2/GrimaRepo/datasets/ILSVRC2012/train/"
data = {'imagenet': ImagenetCScore}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocessing_tr = transforms.Compose([
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
train_data = data[dataset](transform=preprocessing_tr, img_root=img_root, train=True)
test_data = data[dataset](transform=preprocessing_ts, train=True)
arch = "resnet50"

@timing
def train(model, loader, opt, device, criterion):
    loss_meter = AverageMeter()
    model.train()
    total_batches = len(loader)
    for n_batch, (index, x, label) in enumerate(loader):
        opt.zero_grad()
        x = x.to(device)
        label = label.to(device)
        logits = model(x)
        bs = x.shape[0]
        loss = criterion(logits, label)
        loss.backward()
        # Update stats
        loss_meter.update(loss.cpu(), bs)
        opt.step()

        print(f"\r {n_batch + 1}/{total_batches}: Loss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f}", end="")

    return model, [loss_meter]

@timing
def test(model, loader, device, criterion):
    loss_meter = AverageMeter()
    total_batches = len(loader)
    model.eval()
    with torch.no_grad():
        for n_batch, (index, x, label) in enumerate(loader):
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            bs = x.shape[0]
            loss = criterion(logits, label)
            # Update stats
            loss_meter.update(loss.cpu(), bs)

            print(f"\r {n_batch + 1}/{total_batches}: Loss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f}", end="")

    return model, [loss_meter]

in_features = {'resnet18': 512, "resnet34": 512, "resnet50": 2048} 
in_channels = {'imagenet': 3}
n_classes = {'imagenet': 1} # regression task
models = {'resnet18': resnet18, "resnet34": resnet34, "resnet50": resnet50} 
optimizers = {'adam': Adam, 'sgd': SGD}
input_dims = {'imagenet': 224*224*3}
optimizer = "sgd"  # "sgd", "adam"
device = 'cuda'
criterion = MSELoss()
lr = 0.001

model = models[arch]()
# change last layer for regression
model.fc = nn.Linear(in_features[arch],1)

opt = optimizers[optimizer](model.parameters(), lr=lr, momentum=0.9)
n_epochs = 10
train_dl = DataLoader(train_data, batch_size=256)
test_dl = DataLoader(test_data, batch_size=512)

model.to(device)
for epoch in range(1, n_epochs + 1):
    print(f"\nTrain Epoch {epoch}")
    model, stats = train(model, train_dl, opt, device, criterion)
    checkpoint(model, stats, epoch, arch, dataset, split="train")
    print(f"\nTest Epoch {epoch}")
    model, stats = test(model, test_dl, device, criterion)
    checkpoint(model, stats, epoch, arch, dataset, split="test")