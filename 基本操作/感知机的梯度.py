import torch

x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())
print(o.shape)
