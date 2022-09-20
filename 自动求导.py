import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

# 在默认情况下，PyTorch会累计梯度，我们需要清楚之前的值
print(type(x))
# x.grad.zero_()
y = x.sum()
y.backward()
x.grad
print(x)


# 即使构建函数的计算图需要通过python控制流，我们仍然可以计算得到变量的梯度
def f(a):
    b = a * 2
    while b.norm() < 1000 :
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d/a)