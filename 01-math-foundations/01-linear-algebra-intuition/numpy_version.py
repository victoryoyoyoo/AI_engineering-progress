import numpy as np

# NumPy 版本:reference.py 手刻過的向量運算,在 NumPy 裡各自只要一行
# dtype = float:指定成浮點數,避免整數陣列在除法或開根號時出現型態問題
a = np.array([1, 2, 3], dtype = float)
b = np.array([4, 5, 6], dtype = float)

# NumPy 陣列直接用 + 就是「對應位置相加」,對應 Vector.__add__
print(f"a + b = {a + b}")
# np.dot 對兩個 1 維陣列做內積,對應 Vector.dot;結果 1*4+2*5+3*6 = 32
print(f"a dot b = {np.dot(a, b)}")
# np.linalg.norm 預設算 L2 範數,也就是向量長度,對應 Vector.magnitude;sqrt(14) ≈ 3.7417
# linalg = linear algebra(線性代數)子模組,norm 是「範數/長度」
print(f"|a| = {np.linalg.norm(a):.4f}")
# 餘弦相似度 = 內積 / (長度 × 長度),對應 Vector.cosine_similarity;結果約 0.9746
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")
