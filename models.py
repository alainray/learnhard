import torch.nn as nn
import numpy as np
from torch.nn.functional import relu, avg_pool2d


class FC(nn.Module):
    def __init__(self, input_dim=784, layer_sizes=[128, 256], n_classes=10, act=nn.ReLU):
        super(FC, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, layer_sizes[0]))
        self.act = act()
        for i in range(len(layer_sizes) - 1):
            layer_size = layer_sizes[i]
            next_layer = layer_sizes[i + 1]
            layers.append(nn.Linear(layer_size, next_layer))

        # classifier
        cls = nn.Linear(layer_sizes[-1], n_classes)
        layers.append(cls)
        print(layers)
        self.layers = nn.ModuleList(layers)
        print(self.layers)
        self.stats = []

    def forward(self, x):
        f = nn.Flatten()
        x = f(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.act(x)

        return x


class MiniAlexNet(nn.Module):

    def __init__(self, n_classes=10, n_channels=3, bn = False):
        super(MiniAlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.f = nn.Flatten()
        self.conv1 = nn.Conv2d(n_channels, 200, kernel_size=5, stride=2, padding=1)
        self.bn = bn
        if self.bn:
            self.bn1 = nn.BatchNorm2d(200)
            self.bn2 = nn.BatchNorm2d(200)
        #self.conv1 = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(200, 200, kernel_size=5, stride=2, padding=1)
        #
        #self.conv1 = nn.ReLU(inplace=True)
        #self.conv1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(200 * 6 * 6, 384)
        #self.bn3 = nn.BatchNorm1d(384)
        #nn.ReLU(inplace=True),

        self.fc2 = nn.Linear(384, 192)
        #self.bn4 = nn.BatchNorm1d(192)
        #nn.ReLU(inplace=True),

        self.cls = nn.Linear(192, n_classes)

        self.stats = []

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.avgpool(x)
        x = self.f(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.cls(x)

        return x

# Resnet 18
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True):
        super(BasicBlock, self).__init__()
        self.bn = bn
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes, track_running_stats=False))
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                              stride=stride, bias=False))
        self.count = 0

    def forward(self, x):
        if self.bn:
            out = relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = relu(out)
        else:
            out = relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, ncla, nf, n_channels=3, bn=True):
        super(ResNet, self).__init__()
        self.bn = bn
        self.in_planes = nf
        self.conv1 = conv3x3(n_channels, nf * 1, 1)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.linear = nn.Linear(nf * 8 * block.expansion * 4, ncla, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=self.bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = relu(self.bn1(self.conv1(x)))
        else:
            out = relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MiniResNet18(n_classes=10, nf=20, n_channels=3, bn=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, nf, n_channels=n_channels, bn=bn)