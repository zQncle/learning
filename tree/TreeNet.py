import torch
from torch import nn
from torch.utils.data import DataLoader

import create_data
import torchvision


train_dataset = create_data.MyDataset(csv_file='D:/PycharmProjects/learning/tree/treeData/train_.csv',
                                      root_dir='D:/PycharmProjects/learning/tree/treeData',
                                      transform=torchvision.transforms.ToTensor())
# 加载数据集
train_iter = DataLoader(train_dataset, batch_size=128)

for X, y in train_iter:
    print(X.shape)
    print(y.shape)
    break
