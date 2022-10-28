import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

from tree.utils import plot_image

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("运行在：{}", device)
# !nvidia-smi


class MyDataset(Dataset):
    # 构造数据集
    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file: 标签文件的路径.
            root_dir: 所有图片的路径.
            transform: 一系列transform操作
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)  # 返回数据集长度

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])  # 获取图片所在路径
        img = Image.open(img_path).convert('RGB')  # 防止有些图片是RGBA格式

        label_number = self.data_frame.iloc[idx, 2]  # 获取图片的类别标签

        if self.transform:
            img = self.transform(img)

        return img, label_number  # 返回图片和标签


train_dataset = MyDataset(csv_file='D:/PycharmProjects/learning/tree/treeData/train_.csv',
                                      root_dir='D:/PycharmProjects/learning/tree/treeData',
                                      transform=torchvision.transforms.ToTensor())
sample_dataset = MyDataset(csv_file='D:/PycharmProjects/learning/tree/treeData/sample_submission_.csv',
                                       root_dir='D:/PycharmProjects/learning/tree/treeData',
                                       transform=torchvision.transforms.ToTensor())
# 加载数据集
train_iter = DataLoader(train_dataset, batch_size=128, shuffle=True)
sample_iter = DataLoader(sample_dataset, batch_size=128, shuffle=True)
train_len = len(train_dataset)
sample_len = len(sample_dataset)
print("训练集的长度为：{}".format(train_len))
print("验证集的长度为：{}".format(sample_len))


net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 176))

net.to(device)
# 验证数据集
x, y = next(iter(train_iter))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')
# 验证模型的输入输出
X = torch.rand(size=(1, 3, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


# 1.创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 学习率（超参数），自动学习 0.001
learning_rate = 1e-3
# 优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
# 记录网络训练的次数
total_train_step = 0
# 记录网络验证的次数
total_sample_step = 0
# 训练的次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")
# SummaryWriter的使用
'''
1.终端进入conda     conda activate Q
2.运行log文件       tensorboard --logdir=logs_loss
'''

print("开始训练")
for i in range(epoch):
    print("-------------第{}轮训练-------------".format(i+1))
    # 训练
    net.train()
    for data in train_iter:
        # 进行比较，得出loss
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        # 优化器优化模型
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 对其中的参数进行优化
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            # 训练损失
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证步骤开始
    net.eval()
    total_sample_loss = 0
    # 正确数
    total_accuracy = 0

    with torch.no_grad():
        for data in sample_iter:
            # 进行比较，得出loss
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            total_sample_loss += loss.item()
            # 计算准确的个数
            acc = (outputs.argmax(1) == labels).sum()
            total_accuracy += acc

    print("整体验证集上的loss：{}".format(total_sample_loss))
    print("整体验证集上的正确率：{}".format(total_accuracy/sample_len))
    # 验证的损失
    writer.add_scalar("test_loss", total_sample_loss, total_sample_step)
    # 验证集的正确率
    writer.add_scalar("test_accuracy", total_accuracy/sample_len, total_sample_step)
    total_sample_step += 1
    # 保存运行结果
    torch.save(net, "tree_{}.pth".format(i))
    # torch.sava(net.state.dict(), "net_{}.pth".format(i))
    print("模型已保存")

writer.close()