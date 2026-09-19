import numpy as np

# 一層神經網路的前向傳播,NumPy 版本:reference.py 那一大堆 Matrix 運算,這裡幾行就完成
# 公式:output = ReLU(weights @ inputs + bias)

# 一層神經網路的輸入:3 個數字的向量(3維),直立擺放成 (3, 1)
# 巢狀 list 每個內層 list 只有一個數字,所以每個數字自己是一列,shape 才是 (3, 1) 而不是 (3,)
inputs = np.array([[0.5], [0.8], [0.2]])

# 權重矩陣:吃 3 維向量、吐 2 維向量,所以是 (2, 3)
# 每一列是一個「偵測器」:第 0 列 [0.1, -0.3, 0.5] 決定輸出的第 0 個數字怎麼從 3 個輸入組合出來
weights = np.array([
    [0.1, -0.3, 0.5],
    [0.2, 0.4, -0.1]
])

# 偏差(bias):輸出是 2 維,bias 也要是 2 維,(2, 1)
bias = np.array([[0.1], [0.1]])

# weights @ inputs:(2,3) @ (3,1) = (2,1),@ 是矩陣乘法
# + bias:(2,1) + (2,1) 形狀一樣,直接逐項相加(如果一次丟多筆資料,這裡就會觸發 broadcasting)
# np.maximum(0, x):ReLU,逐元素取 0 與該值中較大的,負數變 0、正數不變
# 手算第 0 個輸出:0.1*0.5 + (-0.3)*0.8 + 0.5*0.2 + 0.1 = 0.05 - 0.24 + 0.1 + 0.1 = 0.01
output = np.maximum(0, weights @ inputs + bias)

print(f"Input shape:{inputs.shape}")    # (3, 1)
print(f"Weight shape:{weights.shape}")  # (2, 3)
print(f"Output shape:{output.shape}")   # (2, 1),跟 weights 的列數一致
print(f"Output: {output}")
