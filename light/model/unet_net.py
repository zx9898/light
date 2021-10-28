from __future__ import print_function, division

import torch.utils.data
from model.unet_part import *


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Unet, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = conv3_block(in_channels, filters[0])
        self.down_conv2 = conv3_block(filters[0], filters[1])
        self.down_conv3 = conv3_block(filters[1], filters[2])
        self.down_conv4 = conv3_block(filters[2], filters[3])
        self.down_conv5 = conv3_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.up_conv4 = conv3_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.up_conv3 = conv3_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.up_conv2 = conv3_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.up_conv1 = conv3_block(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc1 = self.down_conv1(x)

        enc2 = self.MaxPool1(enc1)
        enc2 = self.down_conv2(enc2)

        enc3 = self.MaxPool2(enc2)
        enc3 = self.down_conv3(enc3)

        enc4 = self.MaxPool3(enc3)
        enc4 = self.down_conv4(enc4)

        enc5 = self.MaxPool4(enc4)
        enc5 = self.down_conv5(enc5)

        dec5 = self.Up4(enc5)

        dec4 = torch.cat((enc4, dec5), dim=1)
        dec4 = self.up_conv4(dec4)

        dec4 = self.Up3(dec4)

        dec3 = torch.cat((enc3, dec4), dim=1)
        dec3 = self.up_conv3(dec3)

        dec3 = self.Up2(dec3)

        dec2 = torch.cat((enc2, dec3), dim=1)
        dec2 = self.up_conv2(dec2)

        dec2 = self.Up1(dec2)

        dec1 = torch.cat((enc1, dec2), dim=1)
        dec1 = self.up_conv1(dec1)

        out = self.out_conv(dec1)
        return out


class AttUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AttUnet, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = conv3_block(in_channels, filters[0])
        self.down_conv2 = conv3_block(filters[0], filters[1])
        self.down_conv3 = conv3_block(filters[1], filters[2])
        self.down_conv4 = conv3_block(filters[2], filters[3])
        self.down_conv5 = conv3_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.Att4 = Attention_block(filters[3],filters[3],filters[2])
        self.up_conv4 = conv3_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Att3 = Attention_block(filters[2],filters[2],filters[1])
        self.up_conv3 = conv3_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Att2 = Attention_block(filters[1], filters[1], filters[0])
        self.up_conv2 = conv3_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Att1 = Attention_block(filters[0], filters[0], 32)
        self.up_conv1 = conv3_block(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        enc1 = self.down_conv1(x)

        enc2 = self.MaxPool1(enc1)
        enc2 = self.down_conv2(enc2)

        enc3 = self.MaxPool2(enc2)
        enc3 = self.down_conv3(enc3)

        enc4 = self.MaxPool3(enc3)
        enc4 = self.down_conv4(enc4)

        enc5 = self.MaxPool4(enc4)
        enc5 = self.down_conv5(enc5)

        dec5 = self.Up4(enc5)

        att4 = self.Att4(dec5,enc4)
        dec4 = torch.cat((att4, dec5), dim=1)
        dec4 = self.up_conv4(dec4)

        dec4 = self.Up3(dec4)

        att3 = self.Att3(dec4,enc3)
        dec3 = torch.cat((att3, dec4), dim=1)
        dec3 = self.up_conv3(dec3)

        dec3 = self.Up2(dec3)

        att2 = self.Att2(dec3,enc2)
        dec2 = torch.cat((att2, dec3), dim=1)
        dec2 = self.up_conv2(dec2)

        dec2 = self.Up1(dec2)
        att1 = self.Att2(dec2, enc1)
        dec1 = torch.cat((att1, dec2), dim=1)
        dec1 = self.up_conv1(dec1)

        out = self.out_conv(dec1)
        return out



if __name__ == '__main__':
    net = Unet(in_channels=3, num_classes=1)
    # print(net)
    for name,parameters in net.named_parameters():
        print(name,":",parameters.size())