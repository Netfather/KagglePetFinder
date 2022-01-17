# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         plot_logger
# Description:  此文件用于将一系列日志文件中的  训练部分 生成图标 便于查看
# Author:       Administrator
# Date:         2021/5/19
# -------------------------------------------------------------------------------

from matplotlib import pyplot as plt
import gc
import numpy as np
import os

def plot_logger(log_dirs:str,output_dirs:str,figure_title = "default"):
    '''

    :param log_dirs: 日志的文件路径（按照先后顺序输入）
    :param output_dirs: 生成图片的图片输出路径
    :return:
    '''
    total_record = []
    # 根据日志 读入txt文件
    files_list = os.listdir(log_dirs)  # 获得当前文件夹下，所有文件名称的列表
    file_path_list = [os.path.join(log_dirs, img) for img in files_list]  # 将所有文件名称和传入的路径组合起来，成为每个图片的路径
    file_path_list.sort()  # 只要images和labels文件中的文件名字完全一致，那么使用sort保证images和laels一一对应

    for log_pth in file_path_list:
        flag = 0  # 清空一下标志
        print(log_pth)
        with open(log_pth,"r") as log_file:
            while True:
                lines = log_file.readline()  # 整行读取数据
                if not lines:
                    break
                #截断到 ---------------------------------------------------------------------- 这一行
                if (lines.find("-"*30)) != -1:
                    flag = 1
                    continue
                if (flag == 1 and lines != '\n'):
                    per_line_record = []
                    list = (lines.split(" "))   # 3 5   6   9   12  15
                    # for idx,element in enumerate(list):
                    #     print(idx,":", element)
                    # per_line从左到右分别代表rate     iter   epoch | val_loss  lb(lev) | loss0
                    #                       0       1       2          3        4       5
                    # print(list)
                    per_line_record.append(float(list[3]))  # rate
                    # 有星星号的情况
                    per_line_record.append(int(float(list[5][:-1]) * 10000))  # iter 迭代步长
                    per_line_record.append(float(list[6]))  # epoch
                    per_line_record.append(float(list[9]))  # val_loss
                    per_line_record.append(float(list[12]))  # lb(lev())
                    per_line_record.append(float(list[15]))  # loss0

                    # 无星号的情况
                    # per_line_record.append(int(float(list[5]) * 10000))  # iter 迭代步长
                    # per_line_record.append(float(list[7]))  # epoch
                    # per_line_record.append(float(list[10]))  # val_loss
                    # per_line_record.append(float(list[13]))  # lb(lev())
                    # per_line_record.append(float(list[16]))  # loss0


                    # print((lines[66:73])) # 预留为以后debug
                    # print(lines,end="")
                    # 拆分每一行
                    total_record.append(per_line_record)
                    del per_line_record
                    gc.collect()

    # 根据 total_record 数组来进行画图 先转换为np数组
    plot_matrix = np.array(total_record)
    # print(plot_matrix)

    # 如果val_loss 和  lb 为0 说明是断点继续训练之后的值   直接补充上一个值就可以
    # 预处理valloss 消除断点对图的影响
    prev_value_valloss = 0  # 默认给一个较大的就可以了
    for idx,i in enumerate(plot_matrix[::,3]):
        if ( i == 0) :
            i = prev_value_valloss
            plot_matrix[idx,3] = i
        prev_value_valloss = i
    # 预处理lb 距离 消除断点对图的影响

    prev_value_lb = 0.0  # 默认给一个较大的就可以了
    for idx,i in enumerate(plot_matrix[::,4]):
        if ( i == 0) :
            i = prev_value_lb
            plot_matrix[idx, 4] = i
        prev_value_lb = i
    # per_line从左到右分别代表rate     iter   epoch | val_loss  lb(lev) | loss0
    #                       0       1       2          3        4       5
    plt.clf()
    plt.figure(figsize=(20,6.4))
    plt.subplot(1,3,1)
    train_loss, = plt.plot(plot_matrix[::,1],plot_matrix[::,5],color = 'r')
    val_loss, = plt.plot(plot_matrix[::, 1], plot_matrix[::, 3], color='b')
    plt.legend([train_loss, val_loss], ["trainloss", "valloss"],loc = "best",frameon = False)
    plt.subplot(1,3,2)
    dicescore, = plt.plot(plot_matrix[::, 1], plot_matrix[::, 4], color='g',label = "Dicescore")
    plt.legend(handles = [dicescore])
    plt.subplot(1,3,3)
    learning_rate, = plt.plot(plot_matrix[::, 1], plot_matrix[::, 0], color='r' , label = "Learning_rate")
    plt.legend(handles = [learning_rate])


    plt.savefig(output_dirs)


if __name__ == '__main__':
    pass
