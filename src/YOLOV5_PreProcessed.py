# Copy From https://www.kaggle.com/markwijkhuizen/petfinder-eda-yolov5-obj-detection-tfrecords/notebook#YOLOV5-Object-Detection
# 2022年1月6日 更新：
# Version1： 使用YOLOV5模型 将经过clean的数据中 只含有 pet的部分提取出来。  对于27张无法分辨类别的图片，按照原图提取出来
# 组成 petonly数据集 

import numpy as np
import pandas as pd
from PIL import Image

import glob
import sys
import cv2
import imageio
import joblib
import math
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.notebook import tqdm
import torch
yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

## 离线使用参考如下
# Using cache found in /home/shinewine/.cache/torch/hub/ultralytics_yolov5_master                
# Downloading https://ultralytics.com/assets/Arial.ttf to /home/shinewine/.config/Ultralytics/Arial.ttf...                                                                                                               
                                                        
# Get Image Info
def get_image_info(file_path, plot=False,SAVE = True, save_path = "./yolov5_box"):
    # Read Image
    image = imageio.imread(file_path)
    h, w, c = image.shape   
        
    # Get YOLOV5 results using Test Time Augmentation for better result
    results = yolov5x6_model(image, augment=True)
    
    # Mask for pixels containing pets, initially all set to zero
    pet_pixels = np.zeros(shape=[h, w], dtype=np.uint8)
    
    # Dictionary to Save Image Info
    h, w, _ = image.shape
    image_info = { 
        'n_pets': 0, # Number of pets in the image
        'labels': [], # Label assigned to found objects
        'thresholds': [], # confidence score
        'coords': [], # coordinates of bounding boxes
        'x_min': 0, # minimum x coordinate of pet bounding box
        'x_max': w - 1, # maximum x coordinate of pet bounding box
        'y_min': 0, # minimum y coordinate of pet bounding box
        'y_max': h - 1, # maximum x coordinate of pet bounding box
    }
    
    # Save found pets to draw bounding boxes
    pets_found = []
    cut_x1 = np.PINF
    cut_x2 = np.NINF
    cut_y1 = np.PINF
    cut_y2 = np.NINF
    
    # Save info for each pet
    for x1, y1, x2, y2, treshold, label in results.xyxy[0].cpu().detach().numpy():
        label = results.names[int(label)]
        if label in ['cat', 'dog']:
            image_info['n_pets'] += 1
            image_info['labels'].append(label)
            image_info['thresholds'].append(treshold)
            image_info['coords'].append(tuple([x1, y1, x2, y2]))
            image_info['x_min'] = max(x1, image_info['x_min'])
            image_info['x_max'] = min(x2, image_info['x_max'])
            image_info['y_min'] = max(y1, image_info['y_min'])
            image_info['y_max'] = min(y2, image_info['y_max'])
            cut_x1 = min(x1,cut_x1)
            cut_x2 = max(x2,cut_x2)
            cut_y1 = min(y1,cut_y1)
            cut_y2 = max(y2,cut_y2)

            
            # Set pixels containing pets to 1
            pet_pixels[int(y1):int(y2), int(x1):int(x2)] = 1
            
            # Add found pet
            pets_found.append([x1, x2, y1, y2, label])
                
    # Add Pet Ratio in Image
    image_info['pet_ratio'] = pet_pixels.sum() / (h*w)
    
    if SAVE:
        # 1. 将 图片的 box 部分拿出来
        image_origin = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
        # 对于没有找到宠物的图片 也就是 unknow 部分 直接将原图导出
        if image_info['n_pets'] != 0:
            image_cut = image_origin[ int(cut_y1):int(cut_y2),int(cut_x1):int(cut_x2) ,::]
        else:
            image_cut = image_origin
        os.makedirs(save_path,exist_ok= True)
        output_name = file_path.split("/")[-1].split(".")[0] + "_petonly.jpg"
        cv2.imwrite(os.path.join(save_path,output_name),image_cut)


    return image_info


# Image Info
IMAGES_INFO = {
    'n_pets': [],
    'label': [],
    'coords': [],
    'x_min': [],
    'x_max': [],
    'y_min': [],
    'y_max': [],
    'pet_ratio': [],
}

train = pd.read_csv('/storage/Kaggle_Pet_Finder/clean_data_version1/train.csv')
def get_image_file_path(image_id):
    return f'/storage/Kaggle_Pet_Finder/clean_data_version1/train/{image_id}.jpg'


train['file_path'] = train['Id'].apply(get_image_file_path)

for idx, file_path in tqdm(enumerate(train['file_path'])):
    image_info = get_image_info(file_path, plot=False,SAVE = True,save_path= "./clean_data_onlypets")
    
    IMAGES_INFO['n_pets'].append(image_info['n_pets'])
    IMAGES_INFO['coords'].append(image_info['coords'])
    IMAGES_INFO['x_min'].append(image_info['x_min'])
    IMAGES_INFO['x_max'].append(image_info['x_max'])
    IMAGES_INFO['y_min'].append(image_info['y_min'])
    IMAGES_INFO['y_max'].append(image_info['y_max'])
    IMAGES_INFO['pet_ratio'].append(image_info['pet_ratio'])
    
    # Not Every Image can be Correctly Classified
    labels = image_info['labels']
    if len(set(labels)) == 1: # unanimous label
        IMAGES_INFO['label'].append(labels[0])
    elif len(set(labels)) > 1: # Get label with highest confidence
        IMAGES_INFO['label'].append(labels[0])
    else: # unknown label, yolo could not find pet
        IMAGES_INFO['label'].append('unknown')