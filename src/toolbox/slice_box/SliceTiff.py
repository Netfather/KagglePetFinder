# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         SliceTiff
# Description:  此文件用于 将tif文件切片成指定大小 然后送入原有网络进行测试 并合成为一整个结果  最后导出生成的预测图
# Author:       Administrator
# Date:         2021/4/1
# -------------------------------------------------------------------------------

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
# del train set 否则内存不够
import gc
import os
import albumentations as A
import cv2
import pandas as pd
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import random
import numba
import pathlib
from datetime import datetime
import rasterio  # 由于新图像格式不太一致，使用rasterio会读不出某些图片 因此改为使用tiff. # 更新 tiff会爆内存 因此还是使用rasterio
from rasterio.windows import Window
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import glob
from matplotlib import pyplot as plt
from shutil import copyfile


TIFF_PATH = r"C:\Users\Administrator\Desktop\HandVesselOrigin\Processing"
id = 15
IMAGE_NAME = "Image" + str(id) + ".tif"
OUTPU_NAME = "Image"  + str(id) + ".png"

manual_seed = 23
Sigmoid_Threshold = 0.5

# 设定随机种子，方便复现代码
def set_seeds(seed=manual_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds()

trfm = T.Compose([
    # T.ToPILImage(),
    # T.Resize([cfg.INPUT_NET_SIZE,cfg.INPUT_NET_SIZE]),
    T.ToTensor(),
    T.Normalize([0.614],
                [0.213])
])
def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
        # imagenet
        in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset))
        # encoder_depth=5,
        # decoder_channels=[31, 74],
    )
    return model
model = get_model()

# 从原始图片拆分成不同的区域 并保证最小覆盖 返回的是一个由坐标构成的列表
def make_grid(shape, window=(256,256), min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    if (len(window)  != 2):
        raise ("The window size must be (H * W )!!!")
    x, y = shape
    nx = x // (window[0] - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window[0]
    x2 = (x1 + window[0]).clip(0, x)
    ny = y // (window[1] - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window[1]
    y2 = (y1 + window[1]).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

if __name__ == "__main__":

    # device = t.device('cuda:1')
    device = torch.device('cuda')
    model.to(device)

    # 处理trainloss的metric结果
    model.load_state_dict(torch.load("../ChpAtMinVal_diceloss.pth", map_location='cuda:0'))
    model.eval()

    with rasterio.open(os.path.join(TIFF_PATH, IMAGE_NAME)) as dataset:
        total_image = dataset.read([1])
        print(total_image.shape)
        image_max = np.max(total_image)
        image_min = np.min(total_image)
        preds = np.zeros(dataset.shape, dtype=np.uint8)
        slices = make_grid(dataset.shape, window=total_image.shape[1:], min_overlap=0)

        for index, (slc) in enumerate(tqdm(slices)):
            x1, x2, y1, y2 = slc
            print(x1,x2,y1,y2)
            if dataset.count == 1:  # normal
                image = dataset.read([1],
                                     window=Window.from_slices((x1, x2), (y1, y2)))
                # print(image.shape)
                image = np.squeeze(image)
                image = (image - image_min) / (image_max - image_min) * 255
                image = image.astype(np.uint8)

                img = A.InvertImg(p=1.0)(image=image)["image"]  # 翻转图片为反向
                # 手动在末尾添加一个维度
                img = np.expand_dims(img, axis = -1)
                img = trfm(img)

                with torch.no_grad():
                    img = img.to(device)[None]  # 这里加入的是batch维度 这里的测试是每张图的测试
                    score = model(img)[0][0]

                    # 将预测出来的图片 保存到目录
                    score_sigmoid = score.sigmoid().cpu().numpy()

                    # score_sigmoid = cv2.resize(score_sigmoid, (320, 240))

                    preds[x1:x2, y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)

                    # Windows 系统中用这个
        cv2.imwrite(os.path.join(TIFF_PATH,  OUTPU_NAME),np.where(preds == 1, 255, 0))
                    # Linux系统中用这个
                    # cv2.imwrite(os.path.join(cfg.OUTPUT_PRE_Trainloss, img_path.split("/")[-1]), output_image)

    # 如下代码用于切割图片为各个slices
# with rasterio.open(os.path.join(TIFF_PATH,IMAGE_NAME)) as dataset:
#     slices = make_grid(dataset.shape, window=(240,320), min_overlap=32)
#
#     for index, (slc) in enumerate(tqdm(slices)):
#         x1, x2, y1, y2 = slc
#         if dataset.count == 1:  # normal
#             plt.clf()
#             image = dataset.read([1],
#                                  window=Window.from_slices((x1, x2), (y1, y2)))
#             image = np.moveaxis(image, 0, -1)
#
#             image = (image - np.min(image)) / (np.max(image) - np.min(image))
#             image = np.squeeze(image)
#             print(image.shape)
#             cv2.imwrite(os.path.join(TIFF_PATH,"12_test_{}.jpg".format(index)),image*255)
#             # plt.imshow(image,cmap='gray')
#             # print(np.max(image),np.min(image))
#             # plt.show()
