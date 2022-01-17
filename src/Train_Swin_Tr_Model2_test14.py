# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model4_test12
# Description:
# Author:       Administrator
# Date:         2021/11/18
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model2_test11
# Description:
# Author:       Administrator
# Date:         2021/11/17
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model3_test10
# Description:
# Author:       Administrator
# Date:         2021/11/16
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model1_test9
# Description:
# Author:       Administrator
# Date:         2021/11/15
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model2_test8
# Description:
# Author:       Administrator
# Date:         2021/11/14
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model2_test7
# Description:
# Author:       Administrator
# Date:         2021/11/14
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_Swin_Tr_Model2_test6
# Description:  test6 修正cv和lb的差距问题   在最后的fc层引入meta data
#  test1: 使用随机抽取的 超参数 配合  swin_base_patch4_window7_224_in22k 模型的 基础版
#  test2: 在进行NNI超参数搜索后，根据搜索出来的最优组合
#  test3: 根据最优组合 换用encoder swin_large_patch4_window12_384_in22k 但是这个版本最后一层fc是直接一步收缩到 1通道的，不适合svr_boost使用
#  test4： 根据最优组合 换用 swin_large_patch4_window7_224 这个版本使用了阶梯式的全连接，即 从 15xx维到64 到 32 到 1
#           原以为这样可以适用于svr_boost，但是经过测试，结果依然很糟糕，原因可能是由于特征提取的时候没有融合meta data 中的feature 导致的
#  test5: 使用 swin_large_patch4_window12_384_in22k 改用阶梯式fc， 查看结果
#  test6: 修正了val_trfm的问题，同时使用较为激进的数据增强策略，在模型阶段就融合 meta data 中的feature 以便于未来和XGB等方法融合 查看结果
#  test14:  根据 kaggle 讨论区  改为 32 而不是之前 每个epoch跑完才测试一次
#
# Author:       Administrator
# Date:         2021/11/12
# -------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
# 日志初始化
from shutil import copyfile  # 用于复制文件
from datetime import date
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import timm
import gc
from timeit import default_timer as timer
import json

from toolbox.log_writers.log import get_logger
import toolbox.loss_box.binaray_loss as loss_tool
from toolbox.learning_schdule_box.pytorch_cosin_warmup import CosineAnnealingWarmupRestarts
from toolbox.optimizer_box.radam import RAdam
from toolbox.metric_box.valid_metric import mser_with_logits_numpy,mser_numpy_only_for_mserloss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


# 必要参数

hyper_parameter_group ={
    # 主要环境参数设定
    # test1 到  test7 都是使用这个 随机种子
    "seed": 5623,
    "out_dir": r'../model/Swin_Tr_224/test14',
    "log_file_name": "Swin_Tr_test14",
    "Open_Parral_Training": True,
    "initial_checkpoint": None,
    "no_log": False,   # 注意这个参数只在 NNI_SEARCH中使用 其他时候为False
    "Mixed_Precision_Open": False, # 是否开启混精度  以显著增加训练速度
    # 数据集设定
    "image_resize_resolution": 224,
    # skf n-fold参数设定
    "skf_fold": 10,
    # 模型设定
     #  test1 2 : swin_base_patch4_window7_224_in22k
    # test 3 5 6 7 swin_large_patch4_window12_384_in22k
    # test 4 9  13 swin_large_patch4_window7_224   完成测试
    # test 8 swin_base_patch4_window12_384_in22k
    # test 10 swin_large_patch4_window12_384   完成测试
    # test 12  swin_large_patch4_window7_224_in22k
    "encoder_name" : "swin_large_patch4_window7_224",
    "output_dims": 1,
    # 学习率计划 优化器 相关
    "total_epoch" : 10,
    "batch_size" : 32,
    "cycle_epoch" : 10, #这个参数是指 多少次重启 由于是 迁移学习 因此 这个参数和总训练轮数保持一致即可
    "start_lr" : 2e-5,
    "lr_max_cosin" : 2e-5,
    "lr_min_cosin": 1e-6,
    "lr_gammar_cos": 1.0,
    "lr_warmsteps": 100,
    "optimizer_weight_decay" : 0.001,
    # 类型设定
    "loss_type": "bce", # 支持 bce focal mser 三种
    "optimizer_type": "AdamW" , # 支持 AdamW RAdam Adam 三种
    # excel 中的meta data 信息
    "dense_features": [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ],
}



## 0. 必要初始化 提供必要函数
# 将 Id 包装为 对应的路径
def get_train_file_path(image_id):
    return "../data/train/{}.jpg".format(image_id)

# 设定随机种子，方便复现代码
def set_seeds(seed= hyper_parameter_group["seed"] ):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 打印信息 设定 返回一个 当前信息的字符串 输出
def message(mode='print'):
    if mode == ('print'):
        asterisk = ' '
        loss = batch_loss
    if mode == ('log'):
        asterisk = '*' if iteration % iter_save == 0 else ' '
        loss = train_loss

    text = \
        '%0.7f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
        '%4.3f  %5.2f  | ' % (*valid_loss,) + \
        '%4.3f  %4.3f  | ' % (*loss,) + \
        '%s' % (time_to_str(timer() - start_timer, 'min'))

    return text

# 输出当前所用时间
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)>=1) #we support only one param_group
    lr = lr[0]

    return lr

## setup  ----------------------------------------
# 在output目标路径下 新建一个文件夹名字为 checkpoint
for f in ['checkpoint']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 在output目标路径下 新建一个文件夹名字为 checkpoint_best 用于存储 最优的验证结果
for f in ['checkpoint_best']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 2021年7月14日更新： 在输出目录下 放入 source_code目录 存储每一次运行时候的 源代码赋值一份加入
for f in ['source_code']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)



set_seeds(hyper_parameter_group["seed"])

if hyper_parameter_group["no_log"]:
    stdout = print
else:
    logdir = hyper_parameter_group["out_dir"] + '/log'
    logger = get_logger(logdir,OutputOnConsole = True,log_initial= hyper_parameter_group["log_file_name"],logfilename=hyper_parameter_group["log_file_name"])
    stdout = logger.info

stdout('** total seeds setting **\n')
stdout('manual_seed : \n%s\n' % (hyper_parameter_group["seed"]))
stdout('\n')

stdout('** hyper parameter setting **\n')
stdout('hyper parameter group : \n%s\n' % (hyper_parameter_group))
stdout('\n')

name = os.path.basename(__file__)
if hyper_parameter_group["no_log"] == False:
    stdout('Store Source_Code :')
    stdout('\n')
    copyfile(str(name), hyper_parameter_group["out_dir"] + "/source_code/train_source_code.py")
    stdout('Store CFG file :')
    stdout('\n')
    copyfile("cfg.py", hyper_parameter_group["out_dir"] + "/source_code/cfg_source_code.py")


device = 'cuda'




## 1. Data Loading 导入
total_df = pd.read_csv('../data/train_3407_10folds.csv')

stdout('** load train.csv file **\n')
stdout('length of train.csv : \n%s\n' % (total_df.head()))
stdout('\n')

## 2. 数据增强以及 Dataset定义
train_trfm = A.Compose([
    A.RandomRotate90(),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),

    A.OneOf([
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=1),
        A.RandomBrightnessContrast(brightness_limit=0.4, p=1),
    ], p=0.20),


    A.Resize(height= hyper_parameter_group["image_resize_resolution"],width= hyper_parameter_group["image_resize_resolution"],p = 1)
])

val_trfm = A.Compose([
A.Resize(height= hyper_parameter_group["image_resize_resolution"],width= hyper_parameter_group["image_resize_resolution"],p = 1)
])

stdout('Train Transform  : \n%s\n' % (train_trfm))
stdout('Val Transform : \n%s\n' % (val_trfm))

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

class PetfinderDataset(Dataset):
    def __init__(self, df, transform):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        self._F = df[hyper_parameter_group["dense_features"]].values
        self._transform = transform
        self._as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = get_train_file_path(self._X[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._transform:
            image = self._transform(image = image)['image']
        feature = torch.tensor(self._F[idx,:], dtype=torch.float)
        label =  torch.tensor(self._y[idx],dtype=torch.float)
        return {
            "images":self._as_tensor(image),
            "features":feature,
            "targets":label[None],
                }

## 3. 模型定义

# 如下为Model1 定义
class Swin_Tr_224_Model(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features

        self.fc = nn.Linear(self.n_features, hyper_parameter_group["output_dims"])
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.encoder.patch_embed(x)
        if self.encoder.absolute_pos_embed is not None:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        x = self.encoder.layers(x)
        x = self.encoder.norm(x)  # B L C
        x = self.encoder.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, image):
        feature = self.forward_features(image)
        dropout_features = self.dropout(feature)
        output = self.fc(dropout_features)
        return output

# 模型2 定义
class Swin_Tr_384_Model2(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Identity()
        self.fc = nn.Linear(self.n_features, hyper_parameter_group["output_dims"])
        # self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.encoder.patch_embed(x)
        if self.encoder.absolute_pos_embed is not None:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        x = self.encoder.layers(x)
        x = self.encoder.norm(x)  # B L C
        x = self.encoder.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, image):
        feature = self.forward_features(image)
        # dropout_features = self.dropout(feature)
        output = self.fc(feature)
        return output

# 如下为模型三
class Swin_Tr_224_large_Model3(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Identity()
        self.fc3 = nn.Linear(self.n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc1 = nn.Linear(32, hyper_parameter_group["output_dims"])
        self.dropout3_2 = nn.Dropout(0.4)
        self.dropout2_1 = nn.Dropout(0.2)

    def forward(self, image):
        feature = self.encoder(image)
        fc3_out = self.fc3(feature)
        fc2_out = self.fc2(self.dropout3_2(fc3_out))
        fc1_out = self.fc1(self.dropout2_1(fc2_out))
        return fc1_out

#模型2改版 for test5
class Swin_Tr_384_Model2_test5(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Identity()
        self.fc3 = nn.Linear(self.n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc1 = nn.Linear(32, hyper_parameter_group["output_dims"])
        self.dropout3_2 = nn.Dropout(0.4)
        self.dropout2_1 = nn.Dropout(0.2)

    def forward(self, image):
        feature = self.encoder(image)
        fc3_out = self.fc3(feature)
        fc2_out = self.fc2(self.dropout3_2(fc3_out))
        fc1_out = self.fc1(self.dropout2_1(fc2_out))
        return fc1_out

#模型2改版 for test6  为了便于svr _bosst 的使用
class Swin_Tr_384_Model2_test6(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(140, 64)
        self.fc2 = nn.Linear(64, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = self.dropout(x1)
        x = torch.cat([x, features], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        # x = torch.cat([x, x1, features], dim=1)
        return x

#模型2改版 for test6  为了便于svr _bosst 的使用
class Swin_Tr_384_Model2_test7(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x


class Swin_Tr_384_Model2_base_test8(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x


class Swin_Tr_224_Model1_large_test9(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x

class Swin_Tr_384_Model3_large_test10(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x

class Swin_Tr_384_Model2_large_22k_test11(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x


class Swin_Tr_224_Model4_large_22k_test12(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x

class Swin_Tr_224_Model1_large_test13(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x

class Swin_Tr_224_Model1_large_test14(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(hyper_parameter_group["encoder_name"], pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.encoder.head = nn.Linear(self.n_features, 128)
        self.fc1 = nn.Linear(140, hyper_parameter_group["output_dims"])

    def forward(self, image,features):
        x1 = self.encoder(image)
        x = torch.cat([x1, features], dim=1)
        x = self.fc1(x)

        return x
## 4. Loss 优化器 训练总参数定义

# 本次本质上其实也是二值的一种 因此可以使用原有的 几种loss  这里分别考虑  bce  mse focal  这三种loss
# 暂留 mix 部分


## 5. 验证
# 考察 mse 得分  只测试 mse的得分 只考虑这个
@torch.no_grad()
def validation(model, loader):
    model.eval()
    valid_num = 0
    outputs_logits_list = list()
    truth_list = list()
    for data in loader:
        image = data["images"]
        target = data["targets"]
        feature = data["features"]
        valid_num += image.shape[0]
        image, target,feature = image.to(device), target.to(device),feature.to(device)
        # output = model(image)['out']
        with torch.no_grad():
            output = model(image,feature)
            predict = output.data.sigmoid().cpu().numpy()
            truth = target.data.cpu().numpy()
        # 日期修正  发现问题   应该将结果全部 stack起来 然后再求他们的 指标
        outputs_logits_list.append(predict)
        truth_list.append(truth)
        print('\r %8d / %d  %s' % (valid_num, len(val_data.sampler), time_to_str(timer() - start_timer, 'sec')), end='',
                flush=True)
    # 将 对应的list stack起来
    test_models_logits = np.squeeze(np.concatenate(outputs_logits_list, axis=0), axis=1)
    test_models_truth = np.squeeze(np.concatenate(truth_list, axis=0), axis=1)
    msre_meric = mser_with_logits_numpy(test_models_logits, test_models_truth)

    assert (valid_num == len(val_data.sampler))  # len(valid_loader.dataset))
    del image,target,output,outputs_logits_list,truth_list,test_models_logits,test_models_truth
    gc.collect()
    return [msre_meric,0]

## 6. 训练

stdout('SKF folds  : \n%s\n' % (hyper_parameter_group["skf_fold"]))

# 必要参数保存
# metric_total = np.zeros(shape=[hyper_parameter_group["skf_fold"], hyper_parameter_group["total_epoch"]])  # 10 个 fold 每个 fold 训练10个epoch  多出来的是为了预防不精准导致多训练了几个epoch
metric_total = [[],[],[],[],[],[],[],[],[],[]]

for fold in range(10):

 ##########################################################如下代码 生成 对应的 10_fold csv文件 便于云端进行 svr boost 等一系列训练
#     total_df.loc[val_idx,'kfold'] = fold + 1
#     print(total_df.head(5))
# # 检查一下 total_df 中的 kfold 寻仙
# check_metrics = total_df['kfold']
# cnt = [0,0,0,0,0,0,0,0,0,0]
# print(check_metrics.shape)
# for item in check_metrics:
#     cnt[item - 1] = cnt[item - 1] + 1
# print(cnt)
#
# # 检查完毕 将数据导出为 -10——fold csv
# total_df.to_csv("./train_3407_10folds.csv")

##########################################################如上代码 生成 对应的 10_fold csv文件 便于云端进行 svr boost 等一系列训练

    train_df = total_df[total_df.kfold != (fold + 1)].reset_index(drop=True)
    val_df = total_df.loc[total_df.kfold == (fold + 1)].reset_index(drop=True)
    train_dataset = PetfinderDataset(train_df, transform=train_trfm)
    val_dataset = PetfinderDataset(val_df,transform= val_trfm)

    stdout('Train dataset len : \n%s\n' % len(train_dataset))
    stdout('Valid dataset len : \n%s\n' % len(val_dataset))

    train_data = DataLoader(train_dataset, batch_size=hyper_parameter_group["batch_size"],
                            drop_last=True, shuffle=True, num_workers=4,pin_memory= False)
    val_data = DataLoader(val_dataset, sampler=torch.utils.data.SequentialSampler(val_dataset), batch_size=4,
                          shuffle=False, drop_last=False, num_workers=1,pin_memory= False)


    # 定义模型
    model = Swin_Tr_224_Model1_large_test14(pretrained=True)
    if hyper_parameter_group["Open_Parral_Training"]:
        model = torch.nn.DataParallel(model)
    # stdout('ModelSetting  : \n%s\n' % (model))
    stdout('EncoderName  : \n%s\n' % (hyper_parameter_group["encoder_name"]))
    model.to(device)

    if hyper_parameter_group["initial_checkpoint"] is not None:
        stdout('Initial From Exit Pth  : \n%s\n' % (hyper_parameter_group["initial_checkpoint"]))
        f = torch.load(hyper_parameter_group["initial_checkpoint"], map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        model.load_state_dict(state_dict, strict=True)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    # 计算一下用于 记录日志的必要参数
    # 最大迭代 batchsize 次数
    # 注意 如果这里更换了batchsize 那么 num_iter就不是这样了
    meta_iteration_per_epoch = len(train_data)
    num_iteration = hyper_parameter_group["total_epoch"] * meta_iteration_per_epoch
    # 每 iter_log 打印
    iter_log = meta_iteration_per_epoch
    # 每 iter_valid 次验证
    iter_valid = 32  #  fROM https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/289889  EVRY 32 ITERATIONS  run once
    # 每 iter_save 次保存
    iter_save = 32   #
    # 最大循环步长 每10epoch修正一次
    Cycle_Step = meta_iteration_per_epoch * hyper_parameter_group["cycle_epoch"]
    # 最好 验证metric  这里用的 msre 越小越好
    best_valid_metric = np.PINF  # 全局变量

    # loss 定义
    # 支持 bce focal mser 三种
    if hyper_parameter_group["loss_type"] == "mser":
        crition = loss_tool.RMSELoss().to(device)
    if hyper_parameter_group["loss_type"] == "bce":
        crition = torch.nn.BCEWithLogitsLoss().to(device)
    if hyper_parameter_group["loss_type"] == "focal":
        crition = loss_tool.FocalLoss(alpha=0.80, gamma=2).to(device)

    # 定义学习率计划
    # 支持 AdamW RAdam Adam 三种
    if hyper_parameter_group["optimizer_type"] == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=hyper_parameter_group["start_lr"],
                                      weight_decay=hyper_parameter_group["optimizer_weight_decay"])
    if hyper_parameter_group["optimizer_type"] == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=hyper_parameter_group["start_lr"],
                                     weight_decay=hyper_parameter_group["optimizer_weight_decay"])
    if hyper_parameter_group["optimizer_type"] == "RAdam":
        optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=hyper_parameter_group["start_lr"],
                          weight_decay=hyper_parameter_group["optimizer_weight_decay"])

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=Cycle_Step ,  # 1200张图  6个batchsize  10个epoch 学习率降低0.9
                                              cycle_mult=1.0,
                                              max_lr=hyper_parameter_group["lr_max_cosin"],
                                              min_lr=hyper_parameter_group["lr_min_cosin"],
                                              warmup_steps=hyper_parameter_group["lr_warmsteps"],
                                              gamma=hyper_parameter_group["lr_gammar_cos"])

    stdout('** Flod {} Training Start!! **\n'.format(fold+ 1))
    stdout('   batch_size = %d\n' % (hyper_parameter_group["batch_size"]))
    stdout('    Create Scaler !! **\n')
    stdout('                      |----- VALID ---|---- TRAIN/BATCH --------------\n')
    stdout('rate     iter   epoch | loss  lb(lev) | loss0  loss1  | time          \n')
    stdout('----------------------------------------------------------------------\n')

    # ----
    valid_loss = np.zeros(2, np.float32)
    train_loss = np.zeros(2, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss0 = torch.FloatTensor([0]).to(device).sum()
    loss1 = torch.FloatTensor([0]).to(device).sum()

    metric_total_epoch = 0

    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0

    scaler = GradScaler()
    for epoch_id in range(hyper_parameter_group["total_epoch"]):

        for i, data in enumerate(train_data):

            image = data["images"]
            target = data["targets"]
            feature = data["features"]
            # 由于模型2过大 因此不能每个epoch都保存 这里直接保存表现最好的
            # if (iteration % iter_save == 0):
            #     if hyper_parameter_group["Open_Parral_Training"]:
            #         if iteration != start_iteration:
            #             torch.save({
            #                 'state_dict': model.module.state_dict(),
            #                 'iteration': iteration,
            #                 'epoch': epoch,
            #             }, hyper_parameter_group["out_dir"] + '/checkpoint/{}_model_fold{}.pth'.format(iteration,fold+1))
            #     else:
            #         if iteration != start_iteration:
            #             torch.save({
            #                 'state_dict': model.state_dict(),
            #                 'iteration': iteration,
            #                 'epoch': epoch,
            #             }, hyper_parameter_group["out_dir"] + '/checkpoint/{}_model_fold{}.pth'.format(iteration,fold+1))

            if (iteration % iter_valid == 0):
                if iteration != start_iteration:
                    valid_loss = validation(model, val_data)
                    metric_total[fold].append(valid_loss[0])
                    # metric_total[fold, metric_total_epoch] = valid_loss[0]
                    metric_total_epoch = metric_total_epoch + 1
                    # 2021年7月1日 更新：
                    # 如果当前的valid_loss为最好结果 就额外存储以下到 checkpoint_best中
                    if (best_valid_metric > valid_loss[0]):
                        # 为了节约空间 保存表现最好的参数组
                        if hyper_parameter_group["Open_Parral_Training"]:
                            if iteration != start_iteration:
                                torch.save({
                                    'state_dict': model.module.state_dict(),
                                    'iteration': iteration,
                                    'epoch': epoch_id + 1,
                                }, hyper_parameter_group["out_dir"] + '/checkpoint/model_fold{}_best.pth'.format(
                                    fold + 1))
                        else:
                            if iteration != start_iteration:
                                torch.save({
                                    'state_dict': model.state_dict(),
                                    'iteration': iteration,
                                    'epoch': epoch_id + 1,
                                }, hyper_parameter_group["out_dir"] + '/checkpoint/model_fold{}_best.pth'.format(
                                    fold + 1))

                        with open(hyper_parameter_group["out_dir"] +'/checkpoint_best/' + 'fold{}_best_info.json'.format(fold+1), 'w') as outfile:
                            # 写入 过程中计算得到的每副图片的 TP NP
                            json.dump(
                                {
                                 "iteration": iteration,
                                 'epoch': epoch_id + 1,
                                 "Data_time": date.today().strftime("%d/%m/%Y")
                                 },
                                outfile, indent=1)
                        best_valid_metric = valid_loss[0]

            if (iteration % iter_log == 0):
                if iteration != start_iteration:
                    print('\r', end='', flush=True)
                    stdout(message(mode='log') + '\n')

            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            model.train()

            image, target,feature = image.to(device), target.to(device),feature.to(device)

            optimizer.zero_grad()
            # output = model(image)["out"]

            if hyper_parameter_group["Mixed_Precision_Open"] == True:

                with autocast():
                    output = model(image, feature)
                    loss0 = crition(output, target / 100.0)

                scaler.scale(loss0).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()


            else:
                output = model(image,feature)
                if hyper_parameter_group["loss_type"] != "mser":
                    loss0 = crition(output, target/ 100.0)
                else:
                    loss0 = crition(output, target)

                (loss0).backward()
                optimizer.step()
                scheduler.step()

            # print statistics  --------
            epoch += 1 / len(train_data)
            iteration += 1

            # batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
            batch_loss = np.array([loss0.item(), loss1.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            # 求最近 100 个 batch 的平均 loss
            if iteration % 10 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
    del train_dataset,val_dataset,train_data,val_data,model,optimizer,crition,metric_total_epoch
    torch.cuda.empty_cache()
    gc.collect()

stdout(metric_total)
