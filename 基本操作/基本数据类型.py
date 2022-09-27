import torch
import numpy as np

a = torch.randn(2, 3)
print(a.type())
print(a)
print(type(a))
print(isinstance(a, torch.FloatTensor))

print(isinstance(a, torch.cuda.DoubleTensor))
# a = a.cuda()
# print(isinstance(a, torch.cuda.DoubleTensor))

print(torch.tensor(1.))
print(torch.tensor(1.3))
a = torch.tensor(2.2)
print(a.shape)
print(a.size())

print(torch.tensor([1.1]))
print(torch.tensor([1.1, 2.2]))
print(torch.FloatTensor(1))
print(torch.FloatTensor(2))

a = np.ones(2)
print(a)
print(torch.from_numpy(a))