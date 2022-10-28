import torch.utils.data as Data
from torchvision import transforms
import torchvision
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from torch import nn
import d2l.torch as d2l
from sklearn.model_selection import train_test_split,KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# import ttach as tta


class LeavesSet(Data.Dataset):
    """
        construct the dataset
    """

    def __init__(self, images_path, images_label, transform=None, train=True):
        self.imgs = [
            os.path.join('D:/PycharmProjects/learning/tree/treeData', "".join(image_path)) for
            image_path in images_path]

        # if train dataset : get the appropriate label
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False

        # transform
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.imgs[index]
        pil_img = Image.open(image_path)
        if self.transform:
            transform = self.transform
        else:
            # if not define the transform:default resize the figure(224,224) and ToTensor
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        data = transform(pil_img)
        if self.train:
            image_label = self.labels[index]
            return data, image_label
        else:
            return data

    def __len__(self):
        return len(self.imgs)


# load initial data to dataloader，and encode the label
def load_data_leaves(train_transform=None, test_transform=None):
    train_data = pd.read_csv('D:/PycharmProjects/learning/tree/treeData/train.csv')
    test_data = pd.read_csv('D:/PycharmProjects/learning/tree/treeData/test.csv')

    # encode the train label
    labelencoder = LabelEncoder()
    labelencoder.fit(train_data['label'])
    train_data['label'] = labelencoder.transform(train_data['label'])
    label_map = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    label_inv_map = {v: k for k, v in label_map.items()}

    # get the train data and transorm it as a batch
    train_dataSet = LeavesSet(train_data['image'], train_data['label'], transform=train_transform, train=True)
    test_dataSet = LeavesSet(test_data['image'], images_label=0, transform=test_transform, train=False)

    return (
        train_dataSet,
        test_dataSet,
        label_map,
        label_inv_map,
    )


# define the transform
train_transform = transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224 x 224的新图像
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # 添加随机噪声
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset, test_dataset, label_map, label_inv_map = load_data_leaves(train_transform, test_transform)