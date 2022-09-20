import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y)
print(x * y)
print(x / y)
print(x ** y)

x = torch.arange(4)
print(x)
print(x[3])

print(len(x))
# 创建一个矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
# 转置
print(A.T)
# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)

print(A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a*X).shape)
