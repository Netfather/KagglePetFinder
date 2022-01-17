# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         plot_Multi_ZheXian
# Description:  此文件用于根据给定的数据矩阵 生成满意的多分类折线图 生成效果类似于TNT论文中的Figure3
# Author:       Administrator
# Date:         2021/9/24
# -------------------------------------------------------------------------------
from matplotlib import pyplot as plt
import gc
import numpy as np
import os

### 如下用于生成 TRCUnet中的  不同深度 在 不同TRC数量下的折线比较

# Load data of Figure1
data_1mm = np.array([[1,2,3,4],[81.877,81.903,82.032,82.204]])

data_2mm = np.array([[1,2,3,4],[80.869,81.131,81.312,81.587]])

data_3mm = np.array([[1,2,3,4],[79.107,79.614,79.690,80.126]])

data_4mm = np.array([[1,2,3,4],[77.193,77.892,77.982,78.500]])

data_5mm = np.array([[1,2,3,4],[75.487,76.440,76.487,76.966]])

data_6mm = np.array([[1,2,3,4],[73.943,74.997,75.193,75.482]])

# Load data of Figure2

data_palm = np.array([[1,2,3,4],[86.6044,87.2557,88.41536,90.12442]])

data_wrist = np.array([[1,2,3,4],[84.1812,85.2573,87.4814,88.3956]])

plt.clf()
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
data_1mm_line, = plt.plot(data_1mm[0],data_1mm[1],'rv-')

data_2mm_line, = plt.plot(data_2mm[0],data_2mm[1],'bo--')

data_3mm_line, = plt.plot(data_3mm[0],data_3mm[1],'k^-.')

data_4mm_line, = plt.plot(data_4mm[0],data_4mm[1],'gs:')

data_5mm_line, = plt.plot(data_5mm[0],data_5mm[1],'c|--')

data_6mm_line, = plt.plot(data_6mm[0],data_6mm[1],'mx-.')

plt.legend(handles = [data_1mm_line, data_2mm_line,data_3mm_line,data_4mm_line,data_5mm_line,data_6mm_line],
           labels = ["1mm vessel", "2mm vessel","3mm vessel","4mm vessel","5mm vessel","6mm vessel"],
            #bbox_to_anchor = (0.5, 0., 0.5, 0.5),
            ncol=2,
           loc = "lower right",frameon = True)
# 设定 图的间隔 图的必要注解
# plt.title('(a) DSC (%) v.s. Number of TRC blocks on Simulation DRIVE dataset',y = -0.15)
plt.axis([0.8,4.2,73.0,83.0],'equal')
# 设定 显示哪些横纵坐标
plt.yticks([73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0])
plt.xticks([1.,2.,3.,4.])
plt.xlabel(xlabel= 'The numbers of TRC blocks',fontsize = 12)
plt.ylabel(ylabel = 'Dice score (%)',fontsize = 12)
# plt.axis('equal')
plt.grid(color='0.8',linestyle='-.')

plt.subplot(1,2,2)

data_wrist_line, = plt.plot(data_wrist[0],data_wrist[1],'rv-')

data_palm_line, = plt.plot(data_palm[0],data_palm[1],'bo--')

plt.legend(handles = [data_wrist_line, data_palm_line],
           labels = ["NIR wrist vessel", "NIR palm vessel"],
            #bbox_to_anchor = (0.5, 0., 0.5, 0.5),
            ncol=2,
           loc = "lower right",frameon = True)
# 设定 图的间隔 图的必要注解
# plt.title('(b) DSC (%) v.s. Number of TRC blocks on HV_NIR dataset',y = -0.15)
plt.axis([0.8,4.2,81.0,91.0],'equal')
# 设定 显示哪些横纵坐标
plt.yticks([81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0])
plt.xticks([1.,2.,3.,4.])
plt.xlabel(xlabel= 'The numbers of TRC blocks',fontsize = 12)
plt.ylabel(ylabel = 'Dice score (%)',fontsize = 12)
# plt.axis('equal')
plt.grid(color='0.8',linestyle='-.')

# plt.show()
plt.savefig("./figure3.png")