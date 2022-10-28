import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import create_data
from torch import optim

train_dataset = create_data.MyDataset(csv_file='D:/PycharmProjects/learning/tree/treeData/train_.csv',
                                      root_dir='D:/PycharmProjects/learning/tree/treeData',
                                      transform=torchvision.transforms.ToTensor())
sample_dataset = create_data.MyDataset(csv_file='D:/PycharmProjects/learning/tree/treeData/sample_submission_.csv',
                                       root_dir='D:/PycharmProjects/learning/tree/treeData',
                                       transform=torchvision.transforms.ToTensor())
# 加载数据集
train_iter = DataLoader(train_dataset, batch_size=128, shuffle=False)
sample_iter = DataLoader(sample_dataset, batch_size=128, shuffle=False)
train_len = len(train_dataset)
sample_len = len(sample_dataset)
print("训练集的长度为：{}".format(train_len))
print("验证集的长度为：{}".format(sample_len))


class Residual(nn.Module):

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 176))

# 1.创建损失函数
loss_fn = nn.CrossEntropyLoss()

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


for i in range(epoch):
    print("-------------第{}轮训练-------------".format(i+1))
    # 训练
    net.train()
    for data in train_iter:
        # 进行比较，得出loss
        imgs, labels = data
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
    torch.save(net, "tree_{}.zzq".format(i))
    print("模型已保存")

writer.close()