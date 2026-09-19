"""
Lesson 11: 奇異值分解 (Singular Value Decomposition, SVD)
從power iteration手刻SVD，圖片壓縮/去噪/偽逆矩陣用demo驗證

SVD 的一句話:任何矩陣 A(不管形狀)都能拆成 A = U × Σ × Vᵀ,
幾何上是「旋轉(Vᵀ) → 沿軸縮放(Σ) → 再旋轉(U)」三個動作。
Σ 對角線上的數字是奇異值,由大到小排序,代表「這個方向的重要程度」;
只留前 k 個最大的奇異值,就是最佳的 rank-k 近似(截斷 SVD),壓縮、去噪都靠這招。

本檔案的結構:
    🔴 power_iteration     : 反覆乘矩陣,找出最大特徵值與對應方向(手打過)
    🔴 svd_from_scratch    : 用 power_iteration + deflation 湊出完整 SVD(理解層,沒手打)
    🟡 compress/denoise/pseudoinverse : 直接呼叫 np.linalg.svd 的三個應用
    __main__               : 驗證手刻版與 NumPy 一致,並跑三個 demo
"""
import numpy as np  # 陣列、線性代數(norm、svd、outer、column_stack 等)都靠它


# === 🔴 ===
def power_iteration(M, num_iters=100):
    # power iteration: 找矩陣M最大的特徵值/特徵向量
    # 隨機向量反覆乘上M再normalize，收斂到最大特徵值對應的方向
    # 為什麼會收斂:任何向量 = 「最強方向的成分」+「其他方向的成分」,
    # 每次乘上 M,最強方向的成分放大得最多(倍率 = 最大特徵值),多輪之後其他方向相對小到可以忽略
    n = M.shape[1]  # M 的欄數,也就是向量的維度
    v = np.random.randn(n)             # 隨機起點,shape (n,)
    v = v / np.linalg.norm(v)          # normalize 成長度 1;norm = 向量長度

    for _ in range(num_iters):
        Mv = M @ v                      # 用矩陣「推」一次向量
        v = Mv / np.linalg.norm(Mv)     # 長度重設為 1,避免數字越乘越大溢位;只保留方向

    # Rayleigh quotient:v 已收斂且長度為 1,v @ M @ v 就等於特徵值
    # (M @ v ≈ λ v,再跟 v 內積,因為 v·v = 1,得到 λ)
    # 1 維陣列 @ 矩陣 @ 1 維陣列,NumPy 自動當成橫向量、直向量,不用寫 .T
    eigenvalue = v @ M @ v
    return eigenvalue, v


# === 🔴 ===
def svd_from_scratch(A, k=None):
    # 對A^T A做power iteration找最大特徵值/特徵向量(=第一個奇異值/右奇異向量)
    # 用deflation扣掉這個方向後，重複找下一個，直到湊滿k個或矩陣剩餘部分趨近於0
    # 關係:AᵀA = V Σ² Vᵀ,所以 AᵀA 的特徵向量是 V(右奇異向量),特徵值是奇異值的平方
    m, n = A.shape  # A 是 m×n 矩陣
    if k is None:
        k = min(m, n)  # 最多有 min(m,n) 個非零奇異值

    sigmas = []  # 奇異值 σ
    us = []      # 左奇異向量 u(輸出空間的方向)
    vs = []      # 右奇異向量 v(輸入空間的方向)

    # astype(float):確保是浮點數,避免整數矩陣做扣除運算時型態出問題;copy() 避免改到原矩陣
    A_residual = A.copy().astype(float)  # residual = 還沒被解釋掉的剩餘部分

    for _ in range(k):
        # AᵀA 是 n×n 的對稱矩陣,它的最大特徵值方向就是 A 最強的輸入方向
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        # 特徵值幾乎為 0,代表剩餘矩陣已經沒有內容,提早結束
        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)      # 奇異值 = AᵀA 特徵值的平方根
        # 由 A v = σ u 得 u = A v / σ,u 是對應的左奇異向量(長度自動為 1)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        # deflation:扣掉這一層。σ u vᵀ 是「一層秩 1 矩陣」,
        # np.outer(u, v) 是外積,兩個向量做出矩陣,第 (i,j) 格 = u[i]*v[j]
        # 扣完後 residual 的最強方向就變成下一個奇異值/向量
        A_residual = A_residual - sigma * np.outer(u, v)

    # column_stack 把一堆一維向量並排成矩陣,每個向量一「欄」;us 是空 list 時回傳形狀對的空矩陣
    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)  # 一維陣列,不是完整的 Σ 矩陣(跟 np.linalg.svd 的回傳格式一致)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V  # 注意回傳 V,不是 Vᵀ;NumPy 的 svd 回傳的是 Vt


# === 🟡 ===
def compress_image_svd(image_matrix, k):
    # 截斷SVD(truncated SVD)：只用前k個奇異值/向量重建，達成低秩近似壓縮
    # full_matrices=False:只回傳需要的部分(U 是 m×r、Vt 是 r×n,r = min(m,n)),比較省記憶體
    # S 是一維陣列,由大到小排序;Vt 已經是轉置過的 V
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    # U[:, :k]:取前 k 欄 → (m,k);np.diag(S[:k]):前 k 個奇異值排到對角線 → (k,k);Vt[:k, :]:取前 k 列 → (k,n)
    # 三個相乘 (m,k)@(k,k)@(k,n) = (m,n),回到原本的形狀,但只用了 k 層資訊
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed


# === 🟡 ===
def denoise_svd(noisy_matrix, k):
    # 訊號集中在前幾個奇異值，雜訊分散在所有奇異值，截斷即可去噪
    # 程式碼跟 compress_image_svd 完全一樣,只是使用情境不同:這裡丟掉的小奇異值主要是雜訊
    U, S, Vt = np.linalg.svd(noisy_matrix, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# === 🟡 ===
def pseudoinverse_via_svd(A):
    # Moore-Penrose偽逆矩陣：A+ = V * Sigma+ * U^T
    # Sigma+是把Sigma轉置後，非零項取倒數，零項保持為零
    # 用途:A 不是方陣或不可逆時,仍能求出「最接近」的反矩陣,解最小平方法問題
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # 1.0 / S:每個奇異值取倒數(這裡假設奇異值都不為 0);np.diag 放回對角線
    S_inv = np.diag(1.0 / S)
    # Vt.T 轉回 V;shape (n,r)@(r,r)@(r,m) = (n,m),跟 A 的 shape (m,n) 相反,符合「反矩陣」的形狀
    return Vt.T @ S_inv @ U.T


# === 🟢:之後有空再補,推薦系統(latent factors，user-movie ratings矩陣的低秩補全) ===
# === 🟢:之後有空再補,LSA(term-document矩陣用SVD做語意分群) ===


if __name__ == "__main__":
    print("=== 從零實作SVD vs NumPy ===")
    np.random.seed(42)  # 固定亂數種子,結果可重現
    A = np.random.randn(5, 4)  # 5×4 的常態分布隨機矩陣

    U_ours, S_ours, V_ours = svd_from_scratch(A)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    # np.round(x, 4) 四捨五入到 4 位小數;兩行奇異值應該相同
    print("自己算的奇異值:", np.round(S_ours, 4))
    print("NumPy的奇異值:", np.round(S_np, 4))

    # 重建:U @ Σ @ Vᵀ 應該還原出 A;這裡的 V_ours 是 V,所以要 .T
    A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
    # 兩矩陣相減後取 Frobenius 範數(所有元素平方和開根號),越接近 0 越準
    print(f"重建誤差: {np.linalg.norm(A - A_reconstructed):.8f}")

    print("\n=== 圖片壓縮demo(用隨機矩陣模擬圖片) ===")
    np.random.seed(42)
    rows, cols = 200, 300
    image = np.random.randn(rows, cols)  # 純隨機數字沒有結構,壓縮效果會很差(真實照片才有低秩結構)

    for k in [1, 5, 10, 20, 50]:
        compressed = compress_image_svd(image, k)
        # 相對誤差 = 損失的部分大小 / 原圖大小
        error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
        original_size = rows * cols
        # 儲存 k 組 (U 的一欄、一個奇異值、Vt 的一列),共 k*(rows + cols + 1) 個數字
        compressed_size = k * (rows + cols + 1)
        ratio = compressed_size / original_size
        # {k:>3d}:整數靠右佔 3 格;{ratio:.1%}:轉成百分比、1 位小數
        print(f"k={k:>3d}  相對誤差={error:.4f}  儲存空間={ratio:.1%}")

    print("\n=== 雜訊去除demo ===")
    np.random.seed(42)
    # 乾淨訊號:np.outer 做出秩 1 的矩陣(sin 曲線 × cos 曲線),結構簡單,只需要 1 個奇異值就能表達
    clean = np.outer(
        np.sin(np.linspace(0, 4 * np.pi, 100)),   # np.linspace 產生 100 個均勻分布的角度
        np.cos(np.linspace(0, 2 * np.pi, 80)),
    )
    noise = 0.3 * np.random.randn(100, 80)  # 雜訊分散在所有方向
    noisy = clean + noise
    denoised = denoise_svd(noisy, k=5)      # 只留前 5 個方向,大部分雜訊被丟掉

    noisy_error = np.linalg.norm(noisy - clean)      # 加了雜訊後離乾淨版本多遠
    denoised_error = np.linalg.norm(denoised - clean)  # 去噪後離乾淨版本多遠(應該更小)
    print(f"加噪誤差: {noisy_error:.4f}")
    print(f"去噪後誤差: {denoised_error:.4f}")
    print(f"改善比例: {(1 - denoised_error / noisy_error):.1%}")

    print("\n=== 偽逆矩陣解最小平方法(overdetermined system) ===")
    # 3 個方程式、2 個未知數(直線 y = m*x + c 通過三個點),方程式比未知數多,通常沒有精確解
    A_ls = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    b_ls = np.array([3, 5, 6], dtype=float)

    A_pinv = pseudoinverse_via_svd(A_ls)
    x_svd = A_pinv @ b_ls  # 偽逆矩陣乘上 b,就是最小平方解
    # 三種算法應該得到完全一樣的答案:手刻偽逆、lstsq(專門解最小平方)、pinv(NumPy 內建偽逆)
    # lstsq 回傳一個 tuple,[0] 取解;rcond=None 使用新版預設的容忍值,避免警告
    x_lstsq = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]
    x_pinv = np.linalg.pinv(A_ls) @ b_ls

    print(f"SVD偽逆矩陣解: {x_svd}")
    print(f"np.linalg.lstsq解: {x_lstsq}")
    print(f"np.linalg.pinv解: {x_pinv}")
