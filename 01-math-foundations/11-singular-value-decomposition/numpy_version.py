"""
Lesson 11: 用NumPy內建SVD對照
reference.py 手刻的 power_iteration + deflation,在 NumPy 裡是一行 np.linalg.svd;
這份檔案示範內建函式的回傳格式、偽逆矩陣、條件數。
"""
import numpy as np


def demo_numpy_svd():
    np.random.seed(42)
    A = np.random.randn(5, 4)  # 5×4 隨機矩陣
    # np.linalg.svd 回傳三個東西:U、S、Vt
    #   U  : 左奇異向量矩陣
    #   S  : 奇異值,是「一維陣列」(不是完整的 Σ 矩陣),由大到小排序
    #   Vt : 已經轉置過的 V(所以想拿 V 要寫 Vt.T)
    # full_matrices=False:只回傳需要的部分,U 是 (5,4)、Vt 是 (4,4),比較省記憶體
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print("NumPy奇異值:", np.round(S, 4))
    # 印出三個 shape:U (5,4)、S (4,)、Vt (4,4)
    print("U shape:", U.shape, "S shape:", S.shape, "Vt shape:", Vt.shape)
    return U, S, Vt


def demo_numpy_pinv():
    # 3 個方程式、2 個未知數:通過 (1,3)、(2,5)、(3,6) 三個點的最佳直線 y = m*x + c
    A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    b = np.array([3, 5, 6], dtype=float)
    # np.linalg.pinv 直接算偽逆矩陣(Moore-Penrose),內部就是用 SVD;
    # 偽逆 @ b 得到最小平方解,對應 reference.py 的 pseudoinverse_via_svd
    x = np.linalg.pinv(A) @ b
    print("np.linalg.pinv解:", x)
    return x


def demo_condition_number():
    # SVD直接算條件數(condition number) = sigma_max / sigma_min
    # 條件數越大,矩陣越「敏感」:輸入有微小誤差,輸出會被放大很多倍,數值計算容易失準
    A = np.array([[1000, 0], [0, 0.001]])
    print("np.linalg.cond():", np.linalg.cond(A))  # 內建函式,預設就是用最大/最小奇異值
    U, S, Vt = np.linalg.svd(A)
    # S.max() / S.min() = 1000 / 0.001 = 1e6,跟 np.linalg.cond 結果一致
    print("手動算 sigma_max/sigma_min:", S.max() / S.min())


if __name__ == "__main__":
    demo_numpy_svd()
    demo_numpy_pinv()
    demo_condition_number()
