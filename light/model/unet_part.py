from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# class conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(conv_block, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True))
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class conv1_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class conv2_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv2_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv5_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv5_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv7_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv7_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=1, stride=3, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=1, stride=3, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv9_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv9_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=1, stride=4, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=1, stride=4, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_cat, self).__init__()

        self.conv1 = conv1_block(in_channels, out_channels)
        self.conv3 = conv3_block(in_channels, out_channels)
        self.conv5 = conv5_block(in_channels, out_channels)
        self.conv7 = conv7_block(in_channels, out_channels)
        self.conv9 = conv9_block(in_channels, out_channels)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=1, stride=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        s5 = self.conv5(x)
        x7 = self.conv7(x)
        x9 = self.conv9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x


class conv_cc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_cc, self).__init__()

        self.conv_cc1 = conv1_block(in_channels, out_channels)
        self.conv_cc2 = conv2_block(in_channels, out_channels)
        self.conv_cc3 = conv3_block(in_channels, out_channels)
        self.conv_cc5 = conv5_block(in_channels, out_channels)
        self.conv_cc7 = conv7_block(in_channels, out_channels)
        self.conv_cc9 = conv9_block(in_channels, out_channels)


        self.conv_cc_out =nn.Conv2d(out_channels*2,out_channels,kernel_size=1,stride=1,padding=0,bias=True)


    def forward(self, x):
        cc1 =self.conv_cc1(x)
        cc2 = self.conv_cc2(x)
        cc3 = self.conv_cc3(x)
        cc5 = self.conv_cc5(x)
        cc7 = self.conv_cc7(x)
        cc9= self.conv_cc9(x)

        cc =torch.cat(cc3,cc7)
        cc_out = self.conv_cc_out(cc)

        return cc_out


class Attention_block(nn.Module):
    def __init__(self,in_channels_x,in_channels_g,out_channels):
        super(Attention_block,self).__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(in_channels_x,out_channels,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channels_g,out_channels,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        def forward(self,x,g):
            x1 = self.Wx(x)
            g1 = self.Wg(g)
            psi =self.relu(x1+g1)
            psi = self.psi(psi)
            out = x*psi

            return out







class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(Unet, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv3_block(in_channels, filters[0])
        self.Conv2 = conv3_block(filters[0], filters[1])
        self.Conv3 = conv3_block(filters[1], filters[2])
        self.Conv4 = conv3_block(filters[2], filters[3])
        self.Conv5 = conv3_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.up_conv4 = conv3_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.up_conv3 = conv3_block(filters[3], filters[2])

        self.Up2 = conv3_block(filters[2], filters[1])
        self.up_conv2 = conv3_block(filters[2], filters[1])

        self.Up1 = conv3_block(filters[1], filters[0])
        self.up_conv1 = conv3_block(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc1 = self.Conv1(x)

        enc2 = self.MaxPool1(enc1)
        enc2 = self.Conv2(enc2)

        enc3 = self.MaxPool2(enc2)
        enc3 = self.Conv3(enc3)

        enc4 = self.MaxPool3(enc3)
        enc4 = self.Conv4(enc4)

        enc5 = self.MaxPool3(enc4)
        enc5 = self.Conv4(enc5)

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

        dec1 = torch.cat((enc1,dec2),dim=1)
        dec1 = self.up_conv1(dec1)

        out = self.out_conv(dec1)
        return out

if __name__ == '__main__':
    # net = Unet(in_channels=3, num_classes=1)
    # print(net)
    pass
