import numpy as np

# 一層神經網路的輸入:3 個數字的向量(3維),直立擺放成 (3, 1)
inputs = np.array([[0.5], [0.8], [0.2]])

# 權重矩陣:吃 3 維向量、吐 2 維向量,所以是 (2, 3)
weights = np.array([
    [0.1, -0.3, 0.5],
    [0.2, 0.4, -0.1]
])

# 偏差(bias):輸出是 2 維,bias 也要是 2 維,(2, 1)
bias = np.array([[0.1], [0.1]])

output = np.maximum(0, weights @ inputs + bias)

print(f"Input shape:{inputs.shape}")
print(f"Weight shape:{weights.shape}")
print(f"Output shape:{output.shape}")
print(f"Output: {output}")