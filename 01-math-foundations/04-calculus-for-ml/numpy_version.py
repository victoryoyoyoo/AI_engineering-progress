import numpy as np

# NumPy 版本:把手刻的梯度下降迴圈,換成向量運算,更快更簡潔
# 邏輯完全跟 reference.py 的 demo_linear_regression 一樣:
# predict -> compute loss -> compute gradient -> update weight,只是不用自己寫for迴圈算每筆資料

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

np.random.seed(42)
w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b                  # 一次算完所有點的預測值(向量運算,不用迴圈)
    error = pred - y
    loss = np.mean(error ** 2)        # MSE
    dw = np.mean(2 * error * x)       # loss對w的偏導數
    db = np.mean(2 * error)           # loss對b的偏導數
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
