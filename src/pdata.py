# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         pdata
# Description:  此文件用于数据探查
#  由于数据集的数据比较简单，同时为了方便 K-fold进行调用 本次的数据集 集成部分写在 train文件中
# 此文件只用于 读入csv文件 并进行数据分辨率等的观察
# 2021年11月15日修正： 将文件导出为一个10fold的csv文件 方便后续进行 svr boost 的调试
# Author:       Administrator
# Date:         2021/11/2
# -------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
##################################################################################################################
# def set_seeds(seed= 3407):
#     os.environ["PL_GLOBAL_SEED"] = str(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#
# set_seeds(3407)
# # 1. 读入数据 并进行测试  测试文件路径是否正确 测试csv文件是否正确  测试图片上是否能正确读入
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
#
# def get_train_file_path(image_id):
#     return "../data/train/{}.jpg".format(image_id)
#
# def get_test_file_path(image_id):
#     return "../data/test/{}.jpg".format(image_id)
#
# # 为数据增加一行 表示  每个图片对应的路径位置
# train['file_path'] = train['Id'].apply(get_train_file_path)
# test['file_path'] = test['Id'].apply(get_test_file_path)


# 展示一下 文件是否读入
# print(train.head())
# print(test.head())

##################################################################################################################
# 展示一下  训练集的推荐度分布
# train['Pawpularity'].hist()
#
# plt.figure(figsize=(20, 20))
# row, col = 5, 5
# for i in range(row * col):
#     plt.subplot(col, row, i+1)
#     image = cv2.imread(train.loc[i, 'file_path'])
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     target = train.loc[i, 'Pawpularity']
#     plt.imshow(image)
#     plt.title(f"target: {target}")
# plt.show()

##################################################################################################################
##2. 测试图片的宽高分布
# total_image_path = r"../data/train"
# image_list = os.listdir(total_image_path)
# image_list.sort()
# total_image_x = []  # 记录每张图片的 w
# total_image_y = []  # 记录每张图片的 h

# for idx,image_name in enumerate(image_list):
#     print(idx)
#     image_path = os.path.join(total_image_path,image_name)
#     image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
#     # print(image.shape)
#     total_image_x.append(image.shape[0])
#     total_image_y.append(image.shape[1])

# assert(len(total_image_x) == 9912)
# assert(len(total_image_y) == 9912)
# plt.clf()
# plt.scatter(total_image_x,total_image_y)
# plt.show()
# 探查结果显示 最大像素为 1280 x 1280  绝大多数分辨率分布在 600 x 700的区间，因此  224 x 224 的 swin_tr 结果是否优秀存疑？ 但是初步还是使用
# baseline分辨率以方便进行迭代 并检查代码通过性
# 将图标绘制出来

# 2021年12月31日 更新
# 进阶测试  考察一下  图片的分辨率 是否和分数有显著关系
total_image_path = "/storage/Kaggle_Pet_Finder/data/train"
train_csv = pd.read_csv('/storage/Kaggle_Pet_Finder/data/train.csv')
image_list = os.listdir(total_image_path)
image_list.sort()
total_image_x = []  # 记录每张图片的 w
total_image_y = []  # 记录每张图片的 h
image_score = []

for idx,image_name in tqdm(enumerate(image_list)):
    # print(idx)
    # print(image_name)
    
    image_path = os.path.join(total_image_path,image_name)
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    # print(image.shape)
    total_image_x.append(image.shape[0])
    total_image_y.append(image.shape[1])
    # 获得 这张图片对应的 分数
    # image_id = image_name.split(".")[0].split("_")[0]
    image_id = image_name.split(".")[0]
    image_score.append(train_csv.loc[train_csv.Id == image_id]["Pawpularity"].item())
    # print(image_score)

print(len(total_image_x))
# assert(len(total_image_x) == 9912)
# assert(len(total_image_y) == 9912)
plt.clf()
plt.scatter(total_image_x,total_image_y)
plt.savefig("./resolution_test_orgin.png")
plt.clf()
plt.scatter(total_image_x,image_score)
plt.savefig("./resolution_score_x_orgin.png")
plt.clf()
plt.scatter(total_image_y,image_score)
plt.savefig("./resolution_score_y_orgin.png")
plt.clf()


##################################################################################################################
##3. 检查一下 Kfold方法 返回的值是什么样子的 方便进行训练
# 测试成功

# from sklearn.model_selection import StratifiedKFold
# # 这个方法和 KFOLD方法最大不同在于 这个方法会尽量保证 每个fold的test分布和 原有的整个test分布完全相同
# skf = StratifiedKFold(
#     n_splits=10, shuffle=True, random_state=19970711
# )
#
# for fold,(train_idx,val_idx) in enumerate(skf.split(train['Id'],train['Pawpularity'])):
#     train_df = train.loc[train_idx].reset_index(drop=True)
#     val_df = train.loc[val_idx].reset_index(drop=True)
#     print("Following is {} Fold information".format(fold))
#     print(len(train_df))
#     print(len(val_df))


##################################################################################################################
##4.  检查一下 csv 文件中 取出的paw值 每个fold 是否均衡

# train_df = pd.read_csv('../data/train_3407_10folds.csv')
# plt.figure(figsize= (20,40))
#
# for fold in range(10):
#
#     data = train_df[train_df.kfold != fold + 1].Pawpularity.values
#     plt.subplot(5, 2,fold + 1)
#     # data['Pawpularity'].hist()
#     plt.hist(data)
#     # plt.show()
#     print(data.shape)
# plt.show()
