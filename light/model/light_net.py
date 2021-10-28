import torch
import torchvision
import torch.nn as nn
from model.light_part import *


class lightNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(lightNet, self).__init__()

        self.conv1 = conv_in(in_channels, 64)
        self.aspp = ASPP(64, 64)
        self.conv_2 = conv_mid(64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.aspp(x1)
        x3 = self.conv_last(x2)
        out = self.out_conv(x3)
        return out

if __name__ == '__main__':
    net = lightNet(in_channels=1,num_classes=1)
    print(net)
