# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Feature_Fusion_Vision2
# Description:  特征融合模块的第二个版本
# Author:       Administrator
# Date:         2021/5/28
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
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
class FeatureFusionModule(nn.Module):
    def __init__(self,nin:int,nout:int):
        super(FeatureFusionModule, self).__init__()
        self.conv1_1 = kernel_1_1_conv_bn_relu( nin , nin // 2)
        self.cbam =  SCSEModule(in_channels= nin,reduction= 4)
        self.depthwiseconv3_3 = depthwise_separable_conv( nin, nin // 2)
        self.channelReduction1 = nn.Conv2d(2 * nin, nin, kernel_size=1, stride=1, padding=0)
        self.channelReduction2 = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0)
        #self.bn = nn.BatchNorm2d(nout)
        #self.activtion = nn.ReLU(inplace=True)

    def forward(self,x_skip, x_up):
        p_1 = self.conv1_1(x_skip)
        p_2 = self.cbam(x_skip)
        p_3 = self.depthwiseconv3_3(x_skip)
        # 维度融合
        p = torch.cat([p_1,p_2,p_3],dim= 1)
        x = self.channelReduction1(p)
        x += x_up
        x = self.channelReduction2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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
        self.ffusion = FeatureFusionModule(in_channels // 2, out_channels)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 4)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape)
        # print(x2.shape)
        # 双线性插值上采样之后 将大小不符合的部分做一个padding  将周围区域填充0 然后加入网络中
        # input is CHW
        # 双线性插值之后 这里的大小是一模一样的 不需要修改
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        # x = self.ffusion(x2,x1)
        # print(x.shape)
        # print()
        return self.ffusion(x2,x1)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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
        #self.initialize()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    #
    # def initialize(module):
    #     for m in module.modules():
    #
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)


## 2.

if __name__ == '__main__':
    # 用于测试
    from torchsummary import summary
    model = UNet(1,1)
    model = model.eval()

    summary(model.cuda(),input_size=(1,240,320),batch_size= 1)
    # input_data = torch.randn(size=[2,1,256,256])
    # output  = model(input_data)
    # print(output.shape)



