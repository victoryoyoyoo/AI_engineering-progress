"""
Lesson 5「Use It」對應版本:這堂課的「正式函式庫」是PyTorch(torch.autograd)
"""

import torch


x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

y = torch.relu(x1 * x2 + 1)
y.backward()

print(f"PyTorch autograd算出的梯度: dy/dx1={x1.grad.item()}, dy/dx2={x2.grad.item()}")
print("跟 practice.py 的 Value class(自己手刻的自動微分)算出的 dy/dx1=3.0, dy/dx2=2.0 一致")


def numerical_gradient_check(f, x1, x2, h=1e-7):
    """用Lesson4教過的中央差分公式,獨立驗證上面PyTorch算出來的梯度對不對"""
    df_dx1 = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    df_dx2 = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return df_dx1, df_dx2


def f(x1, x2):
    return max(0.0, x1 * x2 + 1)


grad_x1, grad_x2 = numerical_gradient_check(f, 2.0, 3.0)
print(f"\n數值法獨立驗證: dy/dx1={grad_x1:.4f}, dy/dx2={grad_x2:.4f}")
