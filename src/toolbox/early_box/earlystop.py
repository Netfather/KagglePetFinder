# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         earlystop
# Description:  此文件用于实现一个配合 pytorch实现的 earlystop类
# Author:       Administrator
# Date:         2021/5/20
# -------------------------------------------------------------------------------
import torch
import numpy as np
# 定义及早停止类
# 2021年3月27日修正 增加适配 KFold
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, OpenEarlyStop=True, name="Defalut"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            OpenEarlyStop (bool): 如果为真，表明我们需要开启EarlyStop和自动保存最优监视器功能
                                  如果为假，表明我们只开启最优监视器功能
            name (string): 用于存储的checkpoint是哪个指标的
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor_metric_min = np.PINF
        self.monitor_metric_max = np.NINF
        self.OpenES = OpenEarlyStop
        self.name = name

    def __call__(self, monitor_metric, model, id = 0, mode='min'):
        '''
        此函数用于给定一个检测指标，然后按照mode模式来进行模型的保存和 earlyStop
        :param monitor_metric:  需要监测的模型指标
        :param model: 目前正在训练要保存的模型
        :param mode: 需要检测的最优值是min还是max 默认为min
        :param idx: 提示这时第几个id  一般和KFold联动使用
        :return:
        '''
        if (mode == 'min'):
            score = -monitor_metric

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
            elif score < self.best_score:
                if self.OpenES:
                    self.counter += 1
                    print(f'Message From Early Stop: EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
                self.counter = 0
        elif (mode == 'max'):
            score = monitor_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
            elif score < self.best_score:
                if self.OpenES:
                    self.counter += 1
                    print(f'Message From Early Stop: EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
                self.counter = 0

    def save_checkpoint(self, monitor_metric, model, id, mode='min'):

        '''Saves model when validation loss decrease.'''
        if mode == "min":
            if self.verbose:
                print(
                    f'Message From Early Stop: Monite metric:({self.name}) decreased! ({self.monitor_metric_min:.6f} --> {monitor_metric:.6f}).  Saving model ...')
            # 此处应当可以 修改以适配不同的 存储方法
            torch.save(model.state_dict(), './ChpAtMin' + "_id_" + str(id) + self.name + '.pth')

            self.monitor_metric_min = monitor_metric  # 保存完成后更新保存的最优值
        elif mode == "max":
            if self.verbose:
                print(
                    f'Message From Early Stop: Monite metric:({self.name}) increased! ({self.monitor_metric_max:.6f} --> {monitor_metric:.6f}).  Saving model ...')
            # 此处应当可以 修改以适配不同的 存储方法
            torch.save(model.state_dict(), './ChpAtMax' + "_id_" + str(id) + self.name + '.pth')
            self.monitor_metric_max = monitor_metric  # 保存完成后更新保存的最优值
