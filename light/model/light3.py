import torch
import torchvision
import torch.nn as nn
from model.light_part import *


class lightNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(lightNet, self).__init__()

        self.conv1 = conv_in(in_channels, 64)
        self.downsample = nn.MaxPool2d(kernel_size=2,padding=2)
        self.conv3block = conv3block(64,64)
        self.aspp = ASPP(64, 64)
        self.up1 = upsample1(64,64)
        self.up2 = upsample1(64,64)
        self.conv_2 = conv_mid(128, 64)
        self.convyz=conv3block(128,64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3block(x1)
        x3 = self.aspp(x2)
        y1= self.downsample(x1)
        y2 = self.conv3block(y1)
        y3 = self.aspp(y2)
        # y4 = self.up1(y3)

        #----------------#
        z1 = self.downsample(y1)
        z2 = self.conv3block(z1)
        z3 = self.aspp(z2)
        z4 = self.up2(z3)
        catyz = torch.cat((z4,y3),dim=1)
        catyz_out = self.convyz(catyz)
        #--------------------------------#
        #对catyz结果，128->64后，再对64卷积#
        catyz_x = self.conv3block(catyz_out)
        catyx = self.up1(catyz_x)

        catyx_out = torch.cat((x3,catyx),dim=1)
        x4 = self.conv_2(catyz_out)
        out = self.out_conv(x4)
        return out

if __name__ == '__main__':
    net = lightNet(in_channels=1,num_classes=1)
    print(net)
