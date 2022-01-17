# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Train_NNI_HpSearch
# Description:  此文件用于使用NNI进行超参数搜索，
# 2021年10月21日  V1： #1 搜索 针对 TRCUNet_256特化版本进行搜索
# Author:       Administrator
# Date:         2021/10/21
# -------------------------------------------------------------------------------

import os
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
import logging


from toolbox.log_writers.log import get_logger
import toolbox.loss_box.binaray_loss as loss_tool
from toolbox.learning_schdule_box.pytorch_cosin_warmup import CosineAnnealingWarmupRestarts
from toolbox.optimizer_box.radam import RAdam
from toolbox.metric_box.valid_metric import mser_with_logits_numpy,mser_numpy_only_for_mserloss


import nni
from nni.utils import merge_parameter


load_pth  = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
manual_seed = 19970711  #固定住每一次的seed
_logger = logging.getLogger('PetFinder_ML')
image_resize_resolution = 224
skf_fold = 10
encoder_name = "swin_base_patch4_window7_224_in22k"
total_epoch = 10
cycle_epoch = 10 # 用于调节重启次数  由于是迁移学习 轮数较少 与总epoch保持一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

# Global Parameter
train_data = None
val_data = None
model = None
optimizer = None
crition = None
crition_mix_1 = None
crition_mix_2 = None
scheduler = None
train_trfm = None
val_trfm = None
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
iter_per_epoch = 0 # 每个epoch代表了多少iter步长


########################################################################################
# 固定seed
def set_seeds(seed=manual_seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


########################################################################################
# 必要功能函数
def loss_fn_train(y_pred, y_true):
    dice = crition_mix_1(y_pred, y_true * 100.0)  # 由于第一项是 msre 因此需要 x100.
    # dicse = loss_dicse(y_pred, y_true)
    # lovaz = loss_lovaz(y_pred, y_true)
    bec = crition_mix_2(y_pred,y_true)
    return 0.6 * dice + 0.4 * bec

# 将 Id 包装为 对应的路径
def get_train_file_path(image_id):
    return "../data/train/{}.jpg".format(image_id)


## 数据集定义
class PetfinderDataset(Dataset):
    def __init__(self, df, transform):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
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
            image = self._transform(image=image)['image']
        label = self._y[idx]
        return self._as_tensor(image), label[None]


## 3. 模型定义
class Swin_Tr_224_Model(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained)
        # 删除掉最后的 head头
        self.n_features = self.encoder.head.in_features
        self.fc = nn.Linear(self.n_features,1)
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
########################################################################################
# 进行数据准备


# Training
def train(epoch,args):
    global train_data
    global val_data
    global model
    global optimizer
    global iter_per_epoch
    global scheduler
    global crition



    print('\nEpoch: %d' % epoch)
    model.train()
    for i, (image, target) in enumerate(train_data):
        image, target = image.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(image)

        if args['loss_type'] == 'mser_loss':
            loss0 = crition(output, target)
        else:
            loss0 = crition(output, target / 100.0)
        (loss0).backward()
        optimizer.step()
        scheduler.step()


 # [bce_with_logits_numpy,bce_score_with_logits_numpy]
@torch.no_grad()
def test(args):
    global best_acc
    global val_data
    global model

    valid_loss = []
    model.eval()
    valid_num = 0

    for image, target in val_data:
        valid_num += image.shape[0]
        image, target = image.to(device), target.float().to(device)
        # output = model(image)['out']
        with torch.no_grad():
            output = model(image)

        predict = output.data.cpu().numpy()
        truth = target.data.cpu().numpy()
    #待修正 这里的loss有计算图泄露
        if args['loss_type'] == 'mser_loss':
            loss_1 = mser_numpy_only_for_mserloss(predict, truth)
            valid_loss.append(loss_1.item())
        else:
            loss_1 = mser_with_logits_numpy(predict, truth)
            valid_loss.append(loss_1.item())

    assert (valid_num == len(val_data.sampler))  # len(valid_loader.dataset))
    del image,target,output,loss_1
    gc.collect()

    msre_val_score = np.array(valid_loss).mean()  # 越小越好
    if (msre_val_score) < best_acc:
        best_acc =  msre_val_score
    return msre_val_score, best_acc



if __name__ == '__main__':
    try:
        set_seeds(seed = manual_seed)
        RCV_CONFIG = nni.get_next_parameter()
        #RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}
        _logger.debug(RCV_CONFIG)

        print(RCV_CONFIG)
        # params = vars(merge_parameter(get_params(), RCV_CONFIG))

        acc = 0.0
        best_acc = 0.0
        metric_total = np.zeros(shape=[10, total_epoch])  # 10 个 fold 每个 fold 训练total_epoch个epoch

        total_df = pd.read_csv('../data/train.csv')
        print('** load train.csv file **\n')
        print('length of train.csv : \n%s\n' % (total_df.head()))
        print('\n')
        # Parameter
        train_trfm = A.Compose([
            A.RandomRotate90(),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomGamma(p=1),
                A.GaussNoise(p=1)
            ], p=0.20),
            A.OneOf([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.4, p=1),
            ], p=0.20),
            A.Resize(height=image_resize_resolution,
                     width=image_resize_resolution, p=1)
        ])

        val_trfm = A.Compose([
            A.Resize(height=image_resize_resolution,
                     width=image_resize_resolution, p=1)
        ])

        skf = StratifiedKFold(
            n_splits=skf_fold, shuffle=True, random_state=manual_seed
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(total_df['Id'], total_df['Pawpularity'])):

            print('==> Training Flod {} data..'.format(fold+1))
            batchsize = RCV_CONFIG["batch_size"]
            # sgd_momentum = args['sgd_momentum']
            optim_weight_decay = RCV_CONFIG['weight_decay']
            # Data
            print('==> Preparing data..')

            train_df = total_df.loc[train_idx].reset_index(drop=True)
            val_df = total_df.loc[val_idx].reset_index(drop=True)
            train_dataset = PetfinderDataset(train_df, transform=train_trfm)
            val_dataset = PetfinderDataset(val_df, transform=val_trfm)

            print(len(train_dataset))
            print(len(val_dataset))
            # train, valid = torch.utils.data.random_split(Handvessle_tarinval, lengths=[train_len, valid_len],
            #                                              generator=torch.Generator().manual_seed(manual_seed))
            # 通过DataLoader将数据集按照batch加载到符合训练参数的 DataLoader
            # 为了使用 num_workers在windows中  必须要把这个定义定义在main中 而且保证这个DataLoadre只会出现一次
            train_data = DataLoader(train_dataset, batch_size=batchsize,
                                    drop_last=True, shuffle=True, num_workers=4, pin_memory=False)
            val_data = DataLoader(val_dataset, sampler=torch.utils.data.SequentialSampler(val_dataset), batch_size=2,
                                  shuffle=False, drop_last=False, num_workers=2, pin_memory=False)

            # Model
            print('==> Building model..')
            # 预留 用于更新所使用的是哪种模型
            # if args['model'] == 'FF_MLA_Vision2':
            #     model = FF3_MLA9_Vision2.UNet(1,1,transformer_drop_rate= tr_drop_rate)
            # if args['model'] == 'FF_MLA_Vision4':
            #     model = FF3_MLA10_Vision4.UNet(1,1,transformer_drop_rate= tr_drop_rate)
            # if args['model'] == 'FF_MLA_Vision5':
            #     model = FF3_MLA10_Vision5.UNet(1,1,transformer_drop_rate= tr_drop_rate)

            # 2021年7月15日更新 ： 用于NNI 超分辨率搜索#8的模型定义
            # if args['model'] == 'FF_MLA_Vision9':
            #     model = FF3_MLA9_Vision9.UNet(1,1,transformer_drop_rate= tr_drop_rate)
            # if args['model'] == 'FF_MLA_Vision10':
            #     model = FF3_MLA10_Vision10.UNet(1,1,transformer_drop_rate= tr_drop_rate)
            model = Swin_Tr_224_Model(pretrained=True)
            # 预留位置用于更新未来的新模型结构
            # model = FF3_MLA10_Vision4.UNet(1, 1, transformer_drop_rate=tr_drop_rate)

            # 模型迁移到显卡上
            model = model.to(device)

            if load_pth is not None:
                model.load_state_dict(torch.load(load_pth, map_location=device)['state_dict'], strict=True)

            # 可选部分: 进行模型参数预加载

            # Critertion
            print('==> Define Criertion and Optimizer')

            if RCV_CONFIG['loss_type'] == 'bce_loss':
                crition = torch.nn.BCEWithLogitsLoss().to(device)
            if RCV_CONFIG['loss_type'] == 'focal_loss':
                crition = loss_tool.FocalLoss(alpha=0.80, gamma=2).to(device)
            if RCV_CONFIG['loss_type'] == 'mser_loss':
                crition = loss_tool.RMSELoss().to(device)

            # Optimizer
            # if args['optimizer'] == 'SGD':
            #     optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=sgd_momentum, weight_decay=5e-4)
            # if args['optimizer'] == 'Adadelta':
            #     optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
            # if args['optimizer'] == 'Adagrad':
            #     optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
            if RCV_CONFIG['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                       weight_decay=optim_weight_decay)
            if RCV_CONFIG['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                        weight_decay=optim_weight_decay)
            if RCV_CONFIG['optimizer'] == 'RAdam':
                optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=RCV_CONFIG['lr'],
                                  weight_decay=optim_weight_decay)

            meta_iteration_per_epoch = int(len(train_dataset) / batchsize)
            num_iteration = total_epoch * meta_iteration_per_epoch
            Cycle_Step = meta_iteration_per_epoch * cycle_epoch

            # scheduler
            scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                      first_cycle_steps=Cycle_Step,  # 50个epoch为一个cycle
                                                      cycle_mult=1.0,
                                                      max_lr=RCV_CONFIG['lr'],
                                                      min_lr= 1e-6,
                                                      warmup_steps=50,
                                                      gamma=1.0)

            for epoch in range(start_epoch, start_epoch+total_epoch):
                train(epoch,RCV_CONFIG)
                acc, best_acc = test(RCV_CONFIG)
                metric_total[fold,epoch] = acc
                print(metric_total)
                if (fold == 9):  # 只有在最后一个fold的时候 才会回报
                    nni.report_intermediate_result(metric_total.mean(axis= 0)[epoch])

        nni.report_final_result(min(metric_total.mean(axis= 0)))
    except Exception as exception:
        _logger.exception(exception)
        raise

