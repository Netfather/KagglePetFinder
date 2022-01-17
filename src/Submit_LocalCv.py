# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Submit_LocalCv
# Description:  此文件用于在本地服务器上对train计算 cv值
# Author:       Administrator
# Date:         2021/11/9
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Submit_Kaggle
# Description:  此文件用于在kaggle上进行提交
# Author:       Administrator
# Date:         2021/11/4
# -------------------------------------------------------------------------------
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import albumentations as A
# 日志初始化
from datetime import date
import timm
import tqdm
import gc
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error
import json
from toolbox.log_writers.log import get_logger

logdir = './'
logger = get_logger(logdir,OutputOnConsole = True,log_initial= "test3_local_cv",logfilename="test3_local_cv")
stdout = logger.info


# 必要参数
# 必要参数
hyper_parameter_group ={
    # 主要环境参数设定
    # test1 到  test7 都是使用这个 随机种子
    "seed": 3407,
    "out_dir": r'../model/Swin_Tr_224/test11',
    "log_file_name": "Swin_Tr_test11",
    "Open_Parral_Training": True,
    "initial_checkpoint": None,
    "no_log": False,   # 注意这个参数只在 NNI_SEARCH中使用 其他时候为False
    # 数据集设定
    "image_resize_resolution": 384,
    # skf n-fold参数设定
    "skf_fold": 10,
    # 模型设定
     #  test1 2 : swin_base_patch4_window7_224_in22k
    # test 3 5 6 7 swin_large_patch4_window12_384_in22k
    # test 4 9  swin_large_patch4_window7_224   完成测试
    # test 8 swin_base_patch4_window12_384_in22k
    # test 10 swin_large_patch4_window12_384   完成测试
    "encoder_name" : "swin_large_patch4_window12_384_in22k",
    "output_dims": 1,
    # 学习率计划 优化器 相关
    "total_epoch" : 10,
    "batch_size" : 32,
    "cycle_epoch" : 10, #这个参数是指 多少次重启 由于是 迁移学习 因此 这个参数和总训练轮数保持一致即可
    "start_lr" : 1e-5,
    "lr_max_cosin" : 1e-5,
    "lr_min_cosin": 1e-6,
    "lr_gammar_cos": 1.0,
    "lr_warmsteps": 50,
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
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
pth_root_dir = r"/storage/Kaggle_Pet_Finder/model/Swin_Tr_224/test11/checkpoint"


## 0. 必要初始化 提供必要函数

# 将 Id 包装为 对应的路径
def get_train_file_path(image_id):
    return "../data/train/{}.jpg".format(image_id)


# 设定随机种子，方便复现代码
def set_seeds(seed=hyper_parameter_group["seed"]):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# helper funciton
def sigmoid_numpy(x): return 1 / (1 + np.exp(-x))


set_seeds(hyper_parameter_group["seed"])
device = 'cuda'

## 1. test 数据导入
total_df = pd.read_csv('../data/train_3407_10folds.csv')

test_trfm = A.Compose([
    A.Resize(height=hyper_parameter_group["image_resize_resolution"],
             width=hyper_parameter_group["image_resize_resolution"], p=1)
])


##2. dataset 定义
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


##3 . 模型定义
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

##4. 开始测试


sum_metrics = 0
for fold in range(10):

    test_df = total_df.loc[total_df.kfold == (fold + 1)].reset_index(drop=True)
    output_matrix = torch.zeros(size=[len(test_df)], dtype=torch.float32)
    test_dataset = PetfinderDataset(test_df, test_trfm)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=1, shuffle=False, pin_memory=True)




    target_numpy = np.array(test_df["Pawpularity"].values)
    # print(target_numpy.shape)

    model = Swin_Tr_384_Model2_large_22k_test11()
    model.to(device)


    print("Now Processing pth {}/{}:".format(fold + 1, 10))

    # 准备相关参数
    outputs_list = list()

    model.load_state_dict(torch.load(os.path.join(pth_root_dir, "model_fold{}_best.pth".format(fold+1)), map_location=device)['state_dict'],
                          strict=True)
    model.eval()
    outputs_logits_list = list()
    truth_list = list()
    for data in test_loader:
        image = data["images"]
        target = data["targets"]
        feature = data["features"]

        image, target, feature = image.to(device), target.to(device), feature.to(device)
        # output = model(image)['out']
        with torch.no_grad():
            output = model(image, feature)
            predict = output.data.sigmoid().cpu().numpy()
            truth = target.data.cpu().numpy()
        # 日期修正  发现问题   应该将结果全部 stack起来 然后再求他们的 指标
        outputs_logits_list.append(predict)
        truth_list.append(truth)
    # 将 对应的list stack起来
    test_models_logits = np.squeeze(np.concatenate(outputs_logits_list, axis=0), axis=1)
    test_models_truth = np.squeeze(np.concatenate(truth_list, axis=0), axis=1)
    msre_meric = np.sqrt(mean_squared_error(test_models_truth, test_models_logits * 100.))

    stdout("Local CV on {} weights is {}".format(fold+1,msre_meric))
    sum_metrics += msre_meric
    del outputs_logits_list,truth_list,test_models_logits,test_models_truth
    gc.collect()

final_output = sum_metrics / 10
stdout("Local CV on 10 Folds weights is {}".format(final_output ))


