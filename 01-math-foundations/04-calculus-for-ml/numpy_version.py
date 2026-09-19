import numpy as np

# NumPy 版本:把手刻的梯度下降迴圈,換成向量運算,更快更簡潔
# 邏輯完全跟 reference.py 的 demo_linear_regression 一樣:
# predict -> compute loss -> compute gradient -> update weight,只是不用自己寫for迴圈算每筆資料

# dtype=float:指定成浮點數,避免整數陣列在後面做乘除時被當成整數運算
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)  # 真實關係 y = 2x + 1

np.random.seed(42)  # 固定亂數種子,讓每次執行的初始值一樣,結果可重現
# np.random.randn() 抽一個標準常態分布(平均 0、標準差 1)的隨機數,當作 w、b 的初始值
w, b = np.random.randn(), np.random.randn()
lr = 0.01  # 學習率:每次更新走多大

for epoch in range(200):
    pred = w * x + b                  # 一次算完所有點的預測值(向量運算,不用迴圈)
    # w * x:純量乘向量 → shape (5,);+ b:再對每個元素加 b
    error = pred - y                  # shape (5,),每筆資料的預測誤差
    loss = np.mean(error ** 2)        # MSE:誤差平方後取平均
    # 手推的偏導數:loss = mean(error²) → dw = mean(2*error*x),db = mean(2*error)
    dw = np.mean(2 * error * x)       # loss對w的偏導數
    db = np.mean(2 * error)           # loss對b的偏導數
    w -= lr * dw                      # 往梯度反方向走一小步,`-=` 等於 w = w - lr*dw
    b -= lr * db
    # 每 40 個 epoch 印一次,加上最後一次,看 loss 一路下降
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={loss:.6f}")

# 訓練完 w 應該接近 2、b 接近 1
print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
