'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from utils import colorful


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128, dropout=False, non_linear_head=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, 128)
        if not non_linear_head:
            self.linear = nn.Linear(512 * block.expansion, low_dim)
        else:
            logging.info(colorful('Using Non Linear Head'))
            self.linear = nn.Sequential(
                        nn.Linear(512 * block.expansion, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, low_dim),
            )


        if self.dropout:
            self.dropout_layer = nn.Dropout(p=0.5)
        # self.linear = nn.Sequential(
        # 	nn.Linear(512*block.expansion, 512),
        # 	nn.Linear(512, low_dim),
        
        # )
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=-1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        if layer == 1:
            return out
        out = self.layer1(out)
        if layer == 2:
            return out
        out = self.layer2(out)
        if layer == 3:
            return out
        out = self.layer3(out)
        if layer == 4:
            return out
        out = self.layer4(out)
        if layer == 5:
            return out
        if self.dropout:
            out = self.dropout_layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if layer == 6:
            return out
        out = self.linear(out)
        # out = self.l2norm(out)
        return out 


    def forward_convs(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        conv1 = out
        out = self.layer1(out)
        conv2 = out
        out = self.layer2(out)
        conv3 = out
        out = self.layer3(out)
        conv4 = out
        out = self.layer4(out)
        conv5 = out
        return conv1, conv2, conv3, conv4, conv5
    


def ResNet18(low_dim=128, dropout=False, non_linear_head=False):
    return ResNet(BasicBlock, [2,2,2,2], low_dim, dropout=dropout, non_linear_head=non_linear_head)

def ResNet34(low_dim=128):
    return ResNet(BasicBlock, [3,4,6,3], low_dim)

def ResNet50(low_dim=128, dropout=False):
    return ResNet(Bottleneck, [3,4,6,3], low_dim, dropout=dropout)

def ResNet101(low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], low_dim)

def ResNet152(low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], low_dim)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
