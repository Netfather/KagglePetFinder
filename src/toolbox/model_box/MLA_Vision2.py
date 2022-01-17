
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         MLA_Vision1
# Description:  根据transformer中自注意力的灵感，将自注意力机制放在decoder部分，自适应的调整不同层过程中应该采取的权重
# 2021年6月2日  V1. 试一下naive的实现，将每层的通道直接坍缩为1 然后stack组成一个[B,4,H,W]的图片，然后做空间层面的自注意力，返回之后与 skip相乘
# 2021年6月5日  V2. V1 版本中 组成的test并没有做sigmoid 导致很可能出现梯度爆炸的情况，这个版本做出修复
# Author:       Administrator
# Date:         2021/6/2
# -------------------------------------------------------------------------------

import torch as t
import torch
import os
import random
import time
import torch.nn as nn
import torch.nn.functional as F


from torchsummary import summary

## 1.最最基础版本的Unet结构
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 双线性插值上采样之后 将大小不符合的部分做一个padding  将周围区域填充0 然后加入网络中
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MLA(nn.Module):
    def __init__(self,w:int,h:int):
        super(MLA, self).__init__()
        self.W = w
        self.H = h
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 4, 1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(4, 1, 1), nn.Sigmoid())
        self.SoftMax = nn.Softmax(dim = 1)


    def forward(self,x):
        # Step 将4个张量维度对齐
        list = []
        for skip in x:
            # 在channel维度做平均
            skip = F.interpolate(skip,size=(self.W,self.H),mode= "bilinear",align_corners= True)
            list.append(   torch.unsqueeze( torch.mean(skip,dim= 1) ,dim= 1))

        test = torch.cat(list,dim= 1)
        # print(test.shape)
        # 通过自注意力将test 的 每个层 每个空间  都自我注意力一下
        # 2021年6月5日 V2. 修复 这里自我注意力得到新的test之后  将test做一下softmax
        test = test * self.cSE(test) + test * self.sSE(test)
        # print(test.shape)   #torch.Size([2, 4, 256, 256])
        # 在channel 维度做 softmax
        test = self.SoftMax(test)
        # print(test.shape)
        # print(torch.unsqueeze(test[:,3,:,:],dim = 1).shape)
        # print((x[3].shape)[2:4])

        new_x4 = nn.AdaptiveAvgPool2d(output_size= (x[3].shape)[2:4])(torch.unsqueeze(test[:,3,:,:],dim = 1))
        new_x3 = nn.AdaptiveAvgPool2d(output_size= (x[2].shape)[2:4])(torch.unsqueeze(test[:,2,:,:],dim = 1))
        new_x2 = nn.AdaptiveAvgPool2d(output_size= (x[1].shape)[2:4])(torch.unsqueeze(test[:,1,:,:],dim = 1))
        new_x1 = (torch.unsqueeze(test[:,0,:,:],dim = 1))
        return [ x[0] * new_x1,x[1]*new_x2,x[2]*new_x3,x[3]*new_x4]

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.mla = MLA(256,256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # print(x4.shape)
        # print(x1.shape)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        # 加入 MLA模块
        x1,x2,x3,x4 = self.mla([x1,x2,x3,x4])
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


## 2.

if __name__ == '__main__':
    # 用于测试
    model = UNet(1,1)
    model = model.eval()

    # summary(model.cuda(),input_size=(1,256,256),batch_size= 1)
    input_data = torch.randn(size=[2,1,256,256])
    output  = model(input_data)
    print(output.shape)



