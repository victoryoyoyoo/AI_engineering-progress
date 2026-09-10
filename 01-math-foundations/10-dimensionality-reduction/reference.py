"""
Lesson 10: 降維 (Dimensionality Reduction)
PCA / Kernel PCA 從零實作，t-SNE / UMAP 用套件跑demo對照
"""
import numpy as np


# === 🔴 ===
class PCA:
    """
    主成分分析(Principal Component Analysis)
    核心邏輯:找出資料變異最大的方向(特徵向量),按變異量(特徵值)排序,只保留前k個方向
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None              # 保留的k個主成分方向(特徵向量)
        self.mean = None                     # 訓練資料的平均值(用來置中)
        self.eigenvalues = None              # 保留的k個特徵值(=各主成分的變異量)
        self.explained_variance_ratio_ = None  # 每個主成分佔總變異量的比例

    def fit(self, X):
        # 1. 置中:每個特徵減去自己的平均值,讓資料以原點為中心
        #    (PCA只關心「資料怎麼變化」,不關心資料原本座落在哪個位置)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 算共變異數矩陣:cov_matrix[i][j] 代表特徵i和特徵j一起變動的程度
        #    對角線是每個特徵自己的變異數
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. 特徵分解:共變異數矩陣是對稱矩陣,用eigh(對稱矩陣專用,比eig穩定快速)
        #    eigenvectors的每一欄是一個方向,eigenvalues是那個方向上的變異量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 依特徵值由大到小排序(eigh預設是由小到大,所以要反過來)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 5. 只留前k個方向(變異量最大的k個)
        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    # === 🟡 ===
    def transform(self, X):
        # 把資料投影到保留的k個主成分方向上:置中後乘上components的轉置
        # 注意用的是fit()存下的self.mean,不是這批X自己的mean,才能跟訓練資料座標系一致
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        # 從降維空間還原回原始維度(有損,因為丟掉的方向資訊回不來)
        return X_reduced @ self.components + self.mean


# === 🟡 ===
def reconstruction_error(X, X_reconstructed):
    """
    還原誤差(MSE):壓縮後又還原,跟原始資料差多少
    """
    return np.mean((X - X_reconstructed) ** 2)


# === 🟡 ===
class KernelPCA:
    """
    核PCA(Kernel PCA):用RBF核函數,在「隱含的高維特徵空間」做PCA
    不用真的算出高維座標(kernel trick),只需要算兩兩資料點的核矩陣
    適合處理標準PCA(線性)處理不了的非線性資料(例如同心圓)
    """

    def __init__(self, n_components, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.X_fit = None
        self.alphas = None       # 核空間裡的「主成分方向」(用alpha係數表示)
        self.lambdas = None      # 對應的特徵值

    def _rbf_kernel(self, X1, X2):
        # RBF核: exp(-gamma * ||x - y||^2)，兩點越近，核值越接近1；越遠越接近0
        sq_dists = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        return np.exp(-self.gamma * sq_dists)

    def fit_transform(self, X):
        self.X_fit = X
        n = X.shape[0]

        # 1. 算核矩陣 K
        K = self._rbf_kernel(X, X)

        # 2. 在特徵空間裡置中(公式跟一般置中不一樣,因為我們沒有顯式座標)
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # 3. 特徵分解核矩陣
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 4. 只留前k個,並normalize(除以sqrt(eigenvalue))
        self.lambdas = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components] / np.sqrt(
            np.maximum(self.lambdas, 1e-12)
        )

        return K_centered @ self.alphas


if __name__ == "__main__":
    # ---------- Step 2: 合成資料驗證PCA ----------
    np.random.seed(42)
    n_samples = 500

    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

    X_synthetic = np.column_stack([x1, x2, x3])

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_synthetic)

    print(f"Original shape: {X_synthetic.shape}")
    print(f"Reduced shape:  {X_reduced.shape}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

    # ---------- 還原誤差 demo ----------
    X_hat = pca.inverse_transform(X_reduced)
    err = reconstruction_error(X_synthetic, X_hat)
    print(f"\nReconstruction error (k=2 of 3 dims): {err:.6f}")

    # ---------- 跟sklearn對照 ----------
    from sklearn.decomposition import PCA as SklearnPCA

    sklearn_pca = SklearnPCA(n_components=2)
    X_sklearn = sklearn_pca.fit_transform(X_synthetic)
    print(f"\nOur PCA explained variance:     {pca.explained_variance_ratio_}")
    print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")
    diff = np.abs(np.abs(X_reduced) - np.abs(X_sklearn))
    print(f"Max absolute difference: {diff.max():.10f}")

    # ---------- Kernel PCA demo:同心圓(標準PCA分不開,kernel PCA可以) ----------
    from sklearn.datasets import make_circles

    X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)

    kpca = KernelPCA(n_components=2, gamma=10)
    X_kpca = kpca.fit_transform(X_circles)
    print(f"\nKernel PCA on concentric circles, output shape: {X_kpca.shape}")

    linear_pca = PCA(n_components=2)
    X_linear = linear_pca.fit_transform(X_circles)
    print(f"Standard PCA explained variance on circles: {linear_pca.explained_variance_ratio_}")
    # 標準PCA對同心圓幾乎沒用(兩個方向變異量差不多,因為圓形資料沒有明顯的線性主軸)

    # ---------- 之後有空再補:t-SNE / UMAP demo,課程3個Exercises尚未做 ----------
