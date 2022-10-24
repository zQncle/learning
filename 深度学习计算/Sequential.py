import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
X = torch.rand(2, 20)
net(X)
print(X)


class MLP(nn.Module):
    '''
    整合上面代码
    从而实现Sequential的功能
    '''
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        '''
        :param X:
        :return:
        '''
        return self.out(F.relu(self.hidden(X)))


X = torch.rand(2, 20)
net = MLP()
print(net(X))