from __future__ import print_function, division

import torch.utils.data
from model.light_part import *


class lightU(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(lightU, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = conv3block(in_channels, filters[0])
        self.down_conv2 = conv3block(filters[0], filters[1])
        self.down_conv3 = conv3block(filters[1], filters[2])

        self.up1 = upsample1(filters[1], filters[0])
        self.up2 = upsample1(filters[2], filters[1])

        self.upconv1 = conv3block(filters[1], filters[0])
        self.upconv2 = conv3block(filters[2], filters[1])

        self.aspp1 = ASPP(filters[0], filters[0])
        self.aspp2 = ASPP(filters[1], filters[1])
        self.aspp3 = ASPP(filters[2], filters[2])

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        encx1 = self.down_conv1(x)
        encx2 = self.aspp1(encx1)  # 64

        ency1 = self.MaxPool1(encx1)
        ency2 = self.down_conv2(ency1)
        ency3 = self.aspp2(ency2)  # 128

        encz1 = self.MaxPool2(ency1)
        encz2 = self.down_conv3(encz1)
        encz3 = self.aspp3(encz2)  # 256

        encz4 = self.up2(encz3)  # 128
        encyz = torch.cat((encz4, ency3), dim=1)  # 256
        encyz1 = self.upconv2(encyz)  # 128
        encyz2 = self.up1(encyz1)
        encxy = torch.cat((encx2, encyz2), dim=1)  # 128

        #---------out----------#
        dec1 = self.upconv1(encxy)
        out = self.out_conv(dec1)
        return out

if __name__ =="__main__":
    net = lightU(1,1)
    for name,parameters in net.named_parameters():
        print(name,":",parameters.size())