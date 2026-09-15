"""
Lesson 11: 用NumPy內建SVD對照
"""
import numpy as np


def demo_numpy_svd():
    np.random.seed(42)
    A = np.random.randn(5, 4)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print("NumPy奇異值:", np.round(S, 4))
    print("U shape:", U.shape, "S shape:", S.shape, "Vt shape:", Vt.shape)
    return U, S, Vt


def demo_numpy_pinv():
    A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    b = np.array([3, 5, 6], dtype=float)
    x = np.linalg.pinv(A) @ b
    print("np.linalg.pinv解:", x)
    return x


def demo_condition_number():
    # SVD直接算條件數(condition number) = sigma_max / sigma_min
    A = np.array([[1000, 0], [0, 0.001]])
    print("np.linalg.cond():", np.linalg.cond(A))
    U, S, Vt = np.linalg.svd(A)
    print("手動算 sigma_max/sigma_min:", S.max() / S.min())


if __name__ == "__main__":
    demo_numpy_svd()
    demo_numpy_pinv()
    demo_condition_number()
