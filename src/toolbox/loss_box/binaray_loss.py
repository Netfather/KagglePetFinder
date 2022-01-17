# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         binaray_loss
# Description:  此文件返回loss
#               支持  focal diceloss softdice lovaz Jacardloss
# Author:       Administrator
# Date:         2021/5/20
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import numpy as np
import torch.cuda.amp as amp

######################################################################


# ====================================================
# MSER Loss  这个loss和之前的不同就是 这个loss是带平方根的
# ====================================================
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# Focal loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# Paper HeKaiming


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


######################################################################
# Dice loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# This is a very common metric on biomedical use
# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,Open_Sigmoid = True):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if Open_Sigmoid:
            inputs = torch.sigmoid(inputs)

        inputs = inputs.float()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1. - dice


######################################################################
# BCEDICE loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# In somtimes: Diceloss is not stable. Often we mix dice and bce
# PyTorch

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,Open_Sigmoid = True,alpha = 0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if Open_Sigmoid:
            inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = alpha * BCE + (1- alpha) * dice_loss

        return Dice_BCE


######################################################################
# Jaccard or ioU loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# In somtimes: This loss is very often used in Natural Image Segmentation
# PyTorch

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,Open_Sigmoid = True):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if Open_Sigmoid:
            inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


######################################################################
# Tversky loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# in the case of α=β=0.5 the Tversky index simplifies to be the same as
# the Dice coefficient, which is also equal to the F1 score.
# With α=β=1, Equation 2 produces Tanimoto coefficient,
# and setting α+β=1 produces the set of Fβ scores.
# Larger βs weigh recall higher than precision (by placing more emphasis on false negatives).
# PyTorch
ALPHA = 0.5
BETA = 0.5


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA,Open_Sigmoid = True):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if Open_Sigmoid:
            inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


######################################################################
# Lovaz Hinge loss
# Copy from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# for semantic segmentation, particularly for multi- class instances.Specifically,
# it sorts predictions by their error before calculating cumulatively how
# each error affects the IoU score.
# Stolen from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
# 对错误进行了基于Pytorch的修正 理论上应该没有问题，但是实际使用可能会在反向传播时失效
class LovaszBinarayLoss(nn.Module):
    def __init__(self):
        super(LovaszBinarayLoss, self).__init__()

    def forward(self,logits,labels,per_image=True, ignore=None):
        """
            Binary Lovasz hinge loss
              logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
              labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
              per_image: compute the loss per image instead of per batch
              ignore: void class id
            """
        if per_image:
            loss = LovaszBinarayLoss.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss

    # --------------------------- BINARY LOSSES ---------------------------

    def lovasz_hinge_flat(self,logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def flatten_binary_scores(self,scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    # --------------------------- HELPER FUNCTIONS ---------------------------
    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def isnan(x):
        return x != x

    @staticmethod
    def mean(l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(LovaszBinarayLoss.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n



def check_all_losses_run():
    label = torch.ones(size=[2, 1, 3, 4])  # 假设有一个 3*4 的全1 的label标记
    logits_1 = torch.randn(size=[2, 1, 3, 4], requires_grad=True)  # 假设网络输出的结果为二分类的 logits值
    logits_2 = logits_1 - 0.1
    logits_3 = logits_1 + 0.1
    # 测试 loss是否正确给定
    dicebce = DiceBCELoss()
    diceloss = DiceLoss()
    focalloss = FocalLoss()
    iouloss = IoULoss()
    lovazbinar = LovaszBinarayLoss()
    terverskay = TverskyLoss()


    loss_1 = dicebce(logits_1, label)
    loss_2 = dicebce(logits_2, label)
    loss_3 = dicebce(logits_3, label)
    print("dicebce:", loss_1, loss_2, loss_3)

    loss_1 = diceloss(logits_1, label)
    loss_2 = diceloss(logits_2, label)
    loss_3 = diceloss(logits_3, label)
    print("diceloss:", loss_1, loss_2, loss_3)

    loss_1 = focalloss(logits_1, label)
    loss_2 = focalloss(logits_2, label)
    loss_3 = focalloss(logits_3, label)
    print("focalloss:", loss_1, loss_2, loss_3)

    loss_1 = iouloss(logits_1, label)
    loss_2 = iouloss(logits_2, label)
    loss_3 = iouloss(logits_3, label)
    print("iouloss:", loss_1, loss_2, loss_3)

    loss_1 = lovazbinar(logits_1, label)
    loss_2 = lovazbinar(logits_2, label)
    loss_3 = lovazbinar(logits_3, label)
    print("lovazbinar:", loss_1, loss_2, loss_3)

    loss_1 = terverskay(logits_1, label)
    loss_2 = terverskay(logits_2, label)
    loss_3 = terverskay(logits_3, label)
    print("terverskay:", loss_1, loss_2, loss_3)



if __name__ == '__main__':
    check_all_losses_run()


