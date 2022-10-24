import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)
# 存储一个张量列表，然后把他们都回内存
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)
# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)

# 将模型的参数存储为一个叫做‘mlp.params’
torch.save(net.state_dict(), 'mlp.params')
# 实例化了原始多层感知机模型的一个备份。直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)

