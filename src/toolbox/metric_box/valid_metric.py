# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         valid_metric
# Description:  此文件用于提供一些常用的 metric的 numpy 做法，避免出现计算图泄露问题
# 2021年7月2日 修正： 修正对于diceloss评分  没有做 0.5 阈值的问题
# Author:       Administrator
# Date:         2021/5/28
# -------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

# helper funciton
def sigmoid(x): return 1/(1 + np.exp(-x))


## bce loss 的numpy做法

def bce_with_logits_numpy(predict,truth):
    assert(predict.shape == truth.shape)

    # 由于是 logtis 还需要做sigmoid
    predict = torch.tensor(predict)
    truth = torch.tensor(truth)
    # predict = torch.sigmoid(predict)

    return F.binary_cross_entropy_with_logits(predict,truth)



## dice score 的 numpy做法
def bce_score_with_logits_numpy(predict,truth):
    assert (predict.shape == truth.shape)

    # 由于是 logtis 还需要做sigmoid
    predict = sigmoid(predict)

    # 2021年7月2日 修正 对 sigmoid做以下0.5阈值判断
    predict = np.where(predict > 0.5 , 1., 0.)

    predicts = predict.reshape(-1)
    truths = truth.reshape(-1)

    intersection = (predicts * truths).sum()
    dice = (2. * intersection) / (predicts.sum() + truths.sum() + 1e-12)
    return 1 - dice

## mser 均方根 metric
def mser_with_logits_numpy(logits,truth):
    assert (logits.shape == truth.shape)

    metrics = np.sqrt(mean_squared_error(truth, logits * 100.))
    return metrics

def mser_numpy_only_for_mserloss(logits,truth):
    assert (logits.shape == truth.shape)
    metrics = torch.sqrt(((truth - logits) ** 2).mean())
    return metrics
