"""
Lesson 11: 奇異值分解 (Singular Value Decomposition, SVD)
從power iteration手刻SVD，圖片壓縮/去噪/偽逆矩陣用demo驗證
"""
import numpy as np


# === 🔴 ===
def power_iteration(M, num_iters=100):
    # power iteration: 找矩陣M最大的特徵值/特徵向量
    # 隨機向量反覆乘上M再normalize，收斂到最大特徵值對應的方向
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v


# === 🔴 ===
def svd_from_scratch(A, k=None):
    # 對A^T A做power iteration找最大特徵值/特徵向量(=第一個奇異值/右奇異向量)
    # 用deflation扣掉這個方向後，重複找下一個，直到湊滿k個或矩陣剩餘部分趨近於0
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V


# === 🟡 ===
def compress_image_svd(image_matrix, k):
    # 截斷SVD(truncated SVD)：只用前k個奇異值/向量重建，達成低秩近似壓縮
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed


# === 🟡 ===
def denoise_svd(noisy_matrix, k):
    # 訊號集中在前幾個奇異值，雜訊分散在所有奇異值，截斷即可去噪
    U, S, Vt = np.linalg.svd(noisy_matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# === 🟡 ===
def pseudoinverse_via_svd(A):
    # Moore-Penrose偽逆矩陣：A+ = V * Sigma+ * U^T
    # Sigma+是把Sigma轉置後，非零項取倒數，零項保持為零
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1.0 / S)
    return Vt.T @ S_inv @ U.T


# === 🟢:之後有空再補,推薦系統(latent factors，user-movie ratings矩陣的低秩補全) ===
# === 🟢:之後有空再補,LSA(term-document矩陣用SVD做語意分群) ===


if __name__ == "__main__":
    print("=== 從零實作SVD vs NumPy ===")
    np.random.seed(42)
    A = np.random.randn(5, 4)

    U_ours, S_ours, V_ours = svd_from_scratch(A)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    print("自己算的奇異值:", np.round(S_ours, 4))
    print("NumPy的奇異值:", np.round(S_np, 4))

    A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
    print(f"重建誤差: {np.linalg.norm(A - A_reconstructed):.8f}")

    print("\n=== 圖片壓縮demo(用隨機矩陣模擬圖片) ===")
    np.random.seed(42)
    rows, cols = 200, 300
    image = np.random.randn(rows, cols)

    for k in [1, 5, 10, 20, 50]:
        compressed = compress_image_svd(image, k)
        error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
        original_size = rows * cols
        compressed_size = k * (rows + cols + 1)
        ratio = compressed_size / original_size
        print(f"k={k:>3d}  相對誤差={error:.4f}  儲存空間={ratio:.1%}")

    print("\n=== 雜訊去除demo ===")
    np.random.seed(42)
    clean = np.outer(
        np.sin(np.linspace(0, 4 * np.pi, 100)),
        np.cos(np.linspace(0, 2 * np.pi, 80)),
    )
    noise = 0.3 * np.random.randn(100, 80)
    noisy = clean + noise
    denoised = denoise_svd(noisy, k=5)

    noisy_error = np.linalg.norm(noisy - clean)
    denoised_error = np.linalg.norm(denoised - clean)
    print(f"加噪誤差: {noisy_error:.4f}")
    print(f"去噪後誤差: {denoised_error:.4f}")
    print(f"改善比例: {(1 - denoised_error / noisy_error):.1%}")

    print("\n=== 偽逆矩陣解最小平方法(overdetermined system) ===")
    A_ls = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    b_ls = np.array([3, 5, 6], dtype=float)

    A_pinv = pseudoinverse_via_svd(A_ls)
    x_svd = A_pinv @ b_ls
    x_lstsq = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]
    x_pinv = np.linalg.pinv(A_ls) @ b_ls

    print(f"SVD偽逆矩陣解: {x_svd}")
    print(f"np.linalg.lstsq解: {x_lstsq}")
    print(f"np.linalg.pinv解: {x_pinv}")
