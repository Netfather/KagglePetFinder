# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         MLA_Vision3
# Description:  根据transformer中自注意力的灵感，将自注意力机制放在decoder部分，自适应的调整不同层过程中应该采取的权重
# 2021年6月2日  V1. 试一下naive的实现，将每层的通道直接坍缩为1 然后stack组成一个[B,4,H,W]的图片，然后做空间层面的自注意力，返回之后与 skip相乘
# 2021年6月5日  V2. V1 版本中 组成的test并没有做sigmoid 导致很可能出现梯度爆炸的情况，这个版本做出修复
# 2021年6月7日  V3. 融合Vit模型在 skip connection中 首先将 跳跃层的特征锚定在最低语义层  1*32*32 然后将这个视为一个patch 变成 4个patch  然后输入到 transformer之中
# Author:       Administrator
# Date:         2021/6/7
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






from itertools import repeat
import collections.abc

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

# 将
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=32, patch_size=32, in_chans=1, embed_dim=32*32, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 这里的 conv2d是一个不可逆的过程 我们不需要不可逆的过程就直接 32*32维度即可
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x)
        # print(x.shape)
        # x =  x.flatten(2).transpose(1, 2)
        x = x.flatten(2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MLA(nn.Module):
    def __init__(self,w:int,h:int,patch_size:int,depth = 4):
        super(MLA, self).__init__()
        self.W = w
        self.H = h

        # self.cSE = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(4, 4, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Sigmoid(),
        # )
        # self.sSE = nn.Sequential(nn.Conv2d(4, 1, 1), nn.Sigmoid())

        self.PatchSize = patch_size
        self.SoftMax = nn.Softmax(dim = 1)

        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.patchembed_x1  = PatchEmbed( img_size=self.PatchSize, patch_size=self.PatchSize, in_chans=1, embed_dim=32*32)
        self.patchembed_x2  = PatchEmbed( img_size=self.PatchSize, patch_size=self.PatchSize, in_chans=1, embed_dim=32*32)
        self.patchembed_x3  = PatchEmbed( img_size=self.PatchSize, patch_size=self.PatchSize, in_chans=1, embed_dim=32*32)
        self.patchembed_x4  = PatchEmbed( img_size=self.PatchSize, patch_size=self.PatchSize, in_chans=1, embed_dim=32*32)

        dpr = [x.item() for x in torch.linspace(0, 0., depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=1024, num_heads=1, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(1024)



    def forward(self,x):
        # Step 将4个张量维度对齐
        list = []
        for skip in x:
            # 在channel维度做平均
            skip = F.interpolate(skip,size=(self.W,self.H),mode= "bilinear",align_corners= True)
            list.append(   torch.unsqueeze( torch.mean(skip,dim= 1) ,dim= 1)    )

        list[0] = self.patchembed_x1(list[0])
        list[1] = self.patchembed_x2(list[1])
        list[2] = self.patchembed_x3(list[2])
        list[3] = self.patchembed_x4(list[3])

        # 将 B * 1 *32 * 32的4个图 变成一整张图

        test = torch.cat(list,dim= 1)
        # print(test.shape)  # torch.Size([2, 4, 1024])
        test = self.blocks(test)
        # print(x.shape)
        test = self.norm(test)
        # 每一层做一下softmax
        test = self.SoftMax(test)

        # 将test中  每个维度都还原为 32 * 32
        B,C,_ = test.shape
        test = test.reshape(shape = [B,C,32,32])

        new_x4 = torch.unsqueeze(test[:,3,:,:],dim = 1)
        new_x3 = F.interpolate(input=torch.unsqueeze(test[:, 2, :, :], dim=1), size=(x[2].shape)[2:4])
        new_x2 = F.interpolate(input=torch.unsqueeze(test[:, 1, :, :], dim=1), size=(x[1].shape)[2:4])
        new_x1 = F.interpolate(input=torch.unsqueeze(test[:, 0, :, :], dim=1), size=(x[0].shape)[2:4])


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
        self.mla = MLA(32,32,32,1)

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



