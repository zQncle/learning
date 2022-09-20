import torch
x = torch.arange(12)
print(x)
# 长度
print(x.shape)
# 数量
print(x.numel())
# 改变张量的形状 三行四列
print(x.reshape(3, 4))
# 创建一些全零函数
print(torch.zeros((2, 3, 4)))
# 创建一些全 1 函数
print(torch.ones((2, 3, 4)))
# 直接赋值
print(torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]]))
# +-*/
x = torch.tensor([1.0, 2, 2, 3])
y = torch.tensor([1.0, 2, 2, 3])
print(x + y)

print(x.sum)
# 广播机制
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a)
print(b)
print(a + b)
# 读取
# x[0:2, :] = 12
# 分配内存
# before = id(Y)
# Y = Y + X
# id(Y) = before

# 转换为 Numpy 张量
# A = X.numpy()
# B = torch.tensor(A)
# print(type(A))