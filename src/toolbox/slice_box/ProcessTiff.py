# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         ProcessTiff
# Description:  此文件用于处理拍摄到的其他tiff图片 此文件将图片重新归一化为0 1 之间，然后将图像拷贝赋值到指定目标文件下
#               此文件读入TIFF_PATH中所有的tif格式文件，归一化到0 1区间，生成的图片id从 start_id 开始
#               然后最后将两个文件统一命名并复制到  TIFF_OUTPUTPATH 文件夹下
# Author:       Administrator
# Date:         2021/3/30
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

TIFF_PATH = r"C:\Users\Administrator\Desktop\HandVesselOrigin\WangJiazhe\XueDi"
start_id = 108  # 表示生成的图片从第几个id开始
TIFF_OUTPUTPATH = r"C:\Users\Administrator\Desktop\HandVesselOrigin\Total"



# 如下实现将tif读入 转为普通灰度图 然后统一根据start_id修改名字 将灰度图和tif都保存到指定目录下
p = pathlib.Path(TIFF_PATH)
for i,filename in enumerate(p.glob("*.tif")):
    print(filename)
    with rasterio.open(filename) as dataset:
        image = dataset.read([1])
        image = np.moveaxis(image, 0, -1)

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        print(image.shape)
        # image = A.Solarize(p=1.0)(image=image)["image"]  # 翻转图片为反向
        image *= 255

        tif_newname = "Image" + str(start_id) + ".tif"
        jpg_newname = "Image" + str(start_id) + ".jpg"


        # print(os.path.join(TIFF_OUTPUTPATH,tif_newname))
        # print(os.path.join(TIFF_OUTPUTPATH, jpg_newname))
        cv2.imwrite(os.path.join(TIFF_OUTPUTPATH, jpg_newname),image)
        # print(np.max(image),np.min(image))
    copyfile(filename, os.path.join(TIFF_OUTPUTPATH,tif_newname))
    start_id+= 1
print(start_id)
