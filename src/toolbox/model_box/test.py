# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         test
# Description:  用于实验一些基本的nn模块
# Author:       Administrator
# Date:         2021/5/27
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision import models

### 模块1  深度可分离卷积 3*3 模块

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,depth_kernel_size = 3,depth_padding = 1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=depth_kernel_size, padding=depth_padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout)
        self.activtion = nn.ReLU(inplace= True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.activtion(out)
        return out

# test part
# depwise_conv_3_3 = depthwise_separable_conv(12,16)
# input = torch.randn(size= [1,12,45,45])
# output = depwise_conv_3_3(input) #torch.Size([1, 16, 45, 45])
# print(output.shape)
# Test OK !

### 模块2  1*1 卷积配合bn和relu

class kernel_1_1_conv_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(kernel_1_1_conv_bn_relu, self).__init__()
        self.conv1_1 = nn.Conv2d(nin, nout, kernel_size=1, padding=0,bias= False)
        self.bn = nn.BatchNorm2d(nout)
        self.activtion = nn.ReLU(inplace= True)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn(out)
        out = self.activtion(out)
        return out

# test part
# depwise_conv_1_1 = kernel_1_1_conv_bn_relu(12,16)
# input = torch.randn(size= [1,12,45,45])
# output = depwise_conv_1_1(input) #torch.Size([1, 16, 45, 45])
# print(output.shape)
# Test OK !


### 模块3  空间自注意力模块
import segmentation_models_pytorch as smp
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

# test part
# cbam = SCSEModule(12,reduction=2)
# input = torch.randn(size= [1,12,45,45])
# output = cbam(input) #torch.Size([1, 12, 45, 45])
# print(output.shape)
# Test OK !

## 测试 timm 0.4.12 中 新的Vit木块

import timm

timm.models.vit_base_patch16_384


