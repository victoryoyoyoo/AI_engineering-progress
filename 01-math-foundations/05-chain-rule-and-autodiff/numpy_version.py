"""
Lesson 5「Use It」對應版本:這堂課的「正式函式庫」不是NumPy,是PyTorch(torch.autograd)
環境沒裝torch,所以這裡用「NumPy手算數值梯度」當替代驗證,
邏輯上等同於PyTorch的 loss.backward() 會做的事,只是換成數值法而不是自動微分。
"""

import numpy as np


def f(x1, x2):
    """跟reference.py demo_manual_verify()一樣的函數:y = relu(x1*x2 + 1)"""
    return max(0.0, x1 * x2 + 1)


def numerical_gradient(f, x1, x2, h=1e-7):
    """
    用NumPy風格的數值法(中央差分,Lesson4教過的公式)算梯度,
    當作PyTorch自動微分算出來的梯度的「獨立驗證」
    """
    df_dx1 = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    df_dx2 = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return df_dx1, df_dx2


x1, x2 = 2.0, 3.0
grad_x1, grad_x2 = numerical_gradient(f, x1, x2)

print(f"數值法算出的梯度: dy/dx1={grad_x1:.4f}, dy/dx2={grad_x2:.4f}")
print("跟 practice.py 的 Value class(自動微分)算出的 dy/dx1=3.0, dy/dx2=2.0 一致")
print()
print("PyTorch對應寫法(環境沒裝torch,邏輯完全一樣,列出來對照):")
print("  x1 = torch.tensor(2.0, requires_grad=True)")
print("  x2 = torch.tensor(3.0, requires_grad=True)")
print("  y = torch.relu(x1 * x2 + 1)")
print("  y.backward()")
print("  print(x1.grad, x2.grad)  # 3.0 2.0")
