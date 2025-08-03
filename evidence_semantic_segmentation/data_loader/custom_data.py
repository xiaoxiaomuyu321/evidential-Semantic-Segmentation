# -*- encoding: utf-8 -*-
'''
@File    :   DataLoade.py
@Time    :   2020/08/01 10:58:51
@Author  :   AngYi
@Contact :   angyi_jq@163.com
@Department   :  QDKD shuli
@description : 创建Dataset类，处理图片，弄成trainloader validloader testloader（OpenCV版）
'''

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import random

random.seed(78)


import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, voc_root, split_txt='train', input_width=256, input_height=256):
        """
        :param voc_root: VOC2012 数据集根目录路径（如：'datasets/VOCdevkit/VOC2012'）
        :param split_txt: 划分文件名（train、val、test）
        """
        super(CustomDataset, self).__init__()

        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.label_dir = os.path.join(voc_root, 'SegmentationClass')
        self.width = input_width
        self.height = input_height

        txt_path = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split_txt}.txt')
        with open(txt_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_id = self.file_list[index]
        img_path = os.path.join(self.image_dir, f'{file_id}.jpg')
        label_path = os.path.join(self.label_dir, f'{file_id}.png')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        img, label = self.train_transform(img, label, crop_size=(self.width, self.height))

        return img, label

    def train_transform(self, image, label, crop_size=(256, 256)):
        image, label = RandomCrop(crop_size)(image, label)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        label = image2label()(label)
        label = torch.from_numpy(label).long()

        return image, label



class RandomCrop(object):
    def __init__(self, size, pad_color=(255, 255, 255)):
        """
        :param size: 目标裁剪尺寸 (height, width)
        :param pad_color: 填充颜色（默认白色）
        """
        self.size = size
        self.pad_color = pad_color

    def __call__(self, img, label):
        h, w, _ = img.shape
        th, tw = self.size

        # --------- Step 1: 如果原图尺寸不足，先进行白边填充 ---------
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            # 图像填充为白色
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=self.pad_color)

            # 标签填充为背景类颜色（黑色）
            label = cv2.copyMakeBorder(label, top, bottom, left, right,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))  # ← 使用背景类颜色

            # 更新填充后的图像大小
            h, w = img.shape[:2]

        # --------- Step 2: 随机裁剪一个 th x tw 区域 ---------
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        img = img[i:i + th, j:j + tw]
        label = label[i:i + th, j:j + tw]

        return img, label



class image2label():
    def __init__(self, num_classes=21):
        classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted plant',
                   'sheep', 'sofa', 'train', 'tv/monitor']

        colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                    [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        self.colormap = colormap[:num_classes]

        cm2lb = np.zeros(256 ** 3, dtype=np.uint8)
        for i, cm in enumerate(self.colormap):
            cm2lb[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        self.cm2lb = cm2lb

    def __call__(self, image):
        image = np.array(image, dtype=np.int64)
        idx = (image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2]
        label = np.array(self.cm2lb[idx], dtype=np.int64)
        return label


class label2image():
    def __init__(self, num_classes=21):
        self.colormap = colormap(256)[:num_classes].astype('uint8')

    def __call__(self, label_pred, label_true):
        pred = self.colormap[label_pred]
        true = self.colormap[label_true]
        return pred, true


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)
        for j in np.arange(8):
            r += (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g += (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b += (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))
        cmap[i, :] = np.array([r, g, b])
    return cmap


# 定义创建数据加载器的函数
def create_dataloader(voc_root, split_txt, input_width, input_height, batch_size, shuffle, num_workers=2, pin_memory=True):
    dataset = CustomDataset(voc_root, split_txt=split_txt, input_width=input_width, input_height=input_height)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

