import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet34, resnet50

def simplecnn():
    return SimpleCNN([32,64,128],1,add_pooling=False)

in_features = {'resnet18': 512, "resnet34": 512, "resnet50": 2048, "simplecnn": 1152} 
in_channels = {'imagenet': 3, "cifar10": 3, "cifar100": 3}
n_classes = {'imagenet': 10, 'cifar10': 10, 'cifar100': 10} # regression task
models = {'resnet18': resnet18, 
          "resnet34": resnet34,
          "resnet50": resnet50,
          "simplecnn": simplecnn} 

def get_model(args):
    pretrained = True if args.pretrained == 1 else False
    freeze = True if args.freeze == 1 else False
    model = models[args.arch](pretrained=pretrained)

    if freeze:
        # Freeze layers!
        for param in model.parameters():
            param.requires_grad = False

    # change last layer for regression between 0 and 1
    if args.label_type == "score":
        model.fc = nn.Linear(in_features[args.arch],1)
    else:
        model.fc = nn.Linear(in_features[args.arch],args.n_bins)
    
    return model

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self, num_channels, N, num_classes=10, add_pooling=False):
        super(SimpleCNN, self).__init__()

        if add_pooling:
            stride=1
        else:
            stride=2

        layer = nn.Sequential()
        layer.add_module('conv1',nn.Conv2d(3, num_channels[0]*N, kernel_size=3, stride=stride))
        layer.add_module('relu1',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool1',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv2',nn.Conv2d(num_channels[0]*N, num_channels[1]*N, kernel_size=3, stride=stride))
        layer.add_module('relu2',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool2',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv3',nn.Conv2d(num_channels[1]*N, num_channels[2]*N, kernel_size=3, stride=stride))
        layer.add_module('relu3',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool3',nn.MaxPool2d(kernel_size=2, stride=1))
        layer.add_module('flatten', nn.Flatten())
        self.features = layer

        self.fc = nn.Sequential(nn.Linear(1152*N, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    N = 2
    x = torch.rand(10,3,32,32)
    model = SimpleCNN([32,64,128],4,add_pooling=False)
    print(model)
    model(x)