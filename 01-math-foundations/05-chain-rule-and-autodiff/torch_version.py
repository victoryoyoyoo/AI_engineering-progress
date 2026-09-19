"""
Lesson 5「Use It」對應版本:這堂課的「正式函式庫」是PyTorch(torch.autograd)
reference.py 手刻的 Value 引擎,在 PyTorch 裡就是 tensor 加上 requires_grad=True,
呼叫 .backward() 之後每個 tensor 的 .grad 屬性就是梯度。
"""

import torch  # PyTorch:深度學習框架,內建自動微分(autograd)


# requires_grad=True:告訴 PyTorch「這個 tensor 要追蹤梯度」,運算過程會自動記成計算圖
# 對照 reference.py:等於 Value(2.0) 這種會記錄來源的數字
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

# y = relu(x1*x2 + 1) = relu(7) = 7;跟 reference.py 的 demo_manual_verify 是同一個計算圖
y = torch.relu(x1 * x2 + 1)
# .backward() 對照 Value.backward():從 y 往回算出每個葉節點的梯度,存進 .grad
y.backward()

# .grad 是 tensor,.item() 把單一元素的 tensor 轉成一般 Python 數字才好印
# 預期 dy/dx1 = x2 = 3.0,dy/dx2 = x1 = 2.0
print(f"PyTorch autograd算出的梯度: dy/dx1={x1.grad.item()}, dy/dx2={x2.grad.item()}")
print("跟 practice.py 的 Value class(自己手刻的自動微分)算出的 dy/dx1=3.0, dy/dx2=2.0 一致")


def numerical_gradient_check(f, x1, x2, h=1e-7):
    """用Lesson4教過的中央差分公式,獨立驗證上面PyTorch算出來的梯度對不對"""
    # 每次只動一個變數(另一個固定),看輸出變化除以 2h,就是對該變數的偏導數
    df_dx1 = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    df_dx2 = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return df_dx1, df_dx2


def f(x1, x2):
    # 純 Python 版本的同一個函數:max(0, x) 就是 relu,不用 torch,所以不追蹤梯度
    return max(0.0, x1 * x2 + 1)


grad_x1, grad_x2 = numerical_gradient_check(f, 2.0, 3.0)
# 數值法的結果應該跟 autograd 一樣(3.0000, 2.0000),互相印證兩種算法都對
print(f"\n數值法獨立驗證: dy/dx1={grad_x1:.4f}, dy/dx2={grad_x2:.4f}")
