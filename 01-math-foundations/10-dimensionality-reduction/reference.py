"""
Lesson 10: 降維 (Dimensionality Reduction)
PCA / Kernel PCA 從零實作，t-SNE / UMAP 用套件跑demo對照

為什麼要降維:資料有幾百、幾千個特徵時,很多特徵其實是重複或高度相關的。
降維是把資料壓到較少的維度,同時盡量保留「資料怎麼變化」的資訊,
用途包括:視覺化(壓到 2D/3D 才畫得出來)、去除雜訊、加速後續模型訓練。

PCA 的直覺:在資料點雲裡找出「變化最大的方向」當第一根新座標軸,
再找與它垂直、變化次大的方向當第二根,依此類推,最後只保留前 k 根軸。
數學上這些方向就是「共變異數矩陣的特徵向量」,變化量就是對應的特徵值。
"""
import numpy as np  # 陣列與線性代數運算(平均、共變異數、特徵分解都靠它)


# === 🔴 ===
class PCA:
    """
    主成分分析(Principal Component Analysis)
    核心邏輯:找出資料變異最大的方向(特徵向量),按變異量(特徵值)排序,只保留前k個方向
    """

    def __init__(self, n_components):
        self.n_components = n_components   # 要保留幾個主成分(降到幾維)
        self.components = None              # 保留的k個主成分方向(特徵向量)
        self.mean = None                     # 訓練資料的平均值(用來置中)
        self.eigenvalues = None              # 保留的k個特徵值(=各主成分的變異量)
        self.explained_variance_ratio_ = None  # 每個主成分佔總變異量的比例

    def fit(self, X):
        # X 的 shape 是 (樣本數 n, 特徵數 d),每一列是一筆資料、每一欄是一個特徵
        # 1. 置中:每個特徵減去自己的平均值,讓資料以原點為中心
        #    (PCA只關心「資料怎麼變化」,不關心資料原本座落在哪個位置)
        # axis=0:沿著「列」的方向取平均,得到每個特徵一個平均值,shape (d,)
        self.mean = np.mean(X, axis=0)
        # (n,d) 減 (d,):broadcasting,每一列都減去同一組平均值
        X_centered = X - self.mean

        # 2. 算共變異數矩陣:cov_matrix[i][j] 代表特徵i和特徵j一起變動的程度
        #    對角線是每個特徵自己的變異數
        # rowvar=False:告訴 np.cov「每一欄是一個變數」(預設是每一列是變數),結果 shape (d,d)
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. 特徵分解:共變異數矩陣是對稱矩陣,用eigh(對稱矩陣專用,比eig穩定快速)
        #    eigenvectors的每一欄是一個方向,eigenvalues是那個方向上的變異量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 依特徵值由大到小排序(eigh預設是由小到大,所以要反過來)
        # np.argsort 回傳「排序後的索引」(由小到大);[::-1] 把 list 倒過來變成由大到小
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        # eigenvectors 的「欄」才是一個向量,所以要用 [:, sorted_idx] 重排欄的順序,
        # 讓每個向量跟自己的特徵值保持配對
        eigenvectors = eigenvectors[:, sorted_idx]

        # 5. 只留前k個方向(變異量最大的k個)
        # [:, :k] 取前 k 欄;.T 轉置,讓 components 的每一「列」是一個主成分,shape (k, d)
        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)  # 全部 d 個方向的變異量總和
        # 每個主成分的變異量 ÷ 總變異量 = 它解釋了多少比例的資料變化
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self  # 回傳自己,可以串接寫成 PCA(2).fit(X).transform(X)

    # === 🟡 ===
    def transform(self, X):
        # 把資料投影到保留的k個主成分方向上:置中後乘上components的轉置
        # 注意用的是fit()存下的self.mean,不是這批X自己的mean,才能跟訓練資料座標系一致
        # shape:(n,d) @ (d,k) = (n,k),也就是每筆資料從 d 維變成 k 維
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        # 訓練並立刻轉換同一批資料,等於 fit 後接 transform
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        # 從降維空間還原回原始維度(有損,因為丟掉的方向資訊回不來)
        # (n,k) @ (k,d) = (n,d),再加回平均值(當初置中時減掉的)
        return X_reduced @ self.components + self.mean


# === 🟡 ===
def reconstruction_error(X, X_reconstructed):
    """
    還原誤差(MSE):壓縮後又還原,跟原始資料差多少
    """
    # 逐元素相減再平方,最後取全部的平均;越小代表降維損失的資訊越少
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
        self.gamma = gamma       # RBF 核的寬度參數:越大,只有很近的點才算相似
        self.X_fit = None
        self.alphas = None       # 核空間裡的「主成分方向」(用alpha係數表示)
        self.lambdas = None      # 對應的特徵值

    def _rbf_kernel(self, X1, X2):
        # RBF核: exp(-gamma * ||x - y||^2)，兩點越近，核值越接近1；越遠越接近0
        # 兩兩距離平方的展開式:||x-y||² = ||x||² + ||y||² - 2 x·y,可以用矩陣運算一次算完全部組合
        sq_dists = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)   # 每個 X1 點的 ||x||²,變成 (n1,1) 直排
            + np.sum(X2 ** 2, axis=1)                 # 每個 X2 點的 ||y||²,shape (n2,),broadcasting 成 (n1,n2)
            - 2 * X1 @ X2.T                           # 內積項,shape (n1,n2)
        )
        return np.exp(-self.gamma * sq_dists)

    def fit_transform(self, X):
        self.X_fit = X          # 記住訓練資料,之後轉換新資料時還會用到
        n = X.shape[0]

        # 1. 算核矩陣 K
        # K[i][j] = 第 i 點與第 j 點的相似度,shape (n,n)
        K = self._rbf_kernel(X, X)

        # 2. 在特徵空間裡置中(公式跟一般置中不一樣,因為沒有顯式座標可以直接減平均)
        # one_n 是每格都是 1/n 的矩陣,用它左乘/右乘等於「取平均」
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # 3. 特徵分解核矩陣
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        sorted_idx = np.argsort(eigenvalues)[::-1]  # 與 PCA 一樣,由大到小排序
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 4. 只留前k個,並normalize(除以sqrt(eigenvalue))
        self.lambdas = eigenvalues[:self.n_components]
        # np.maximum(lambdas, 1e-12):避免特徵值極小或為負(浮點誤差)時,開根號後除以 0
        self.alphas = eigenvectors[:, :self.n_components] / np.sqrt(
            np.maximum(self.lambdas, 1e-12)
        )

        # 降維後的座標 = 置中核矩陣 @ alphas,shape (n, k)
        return K_centered @ self.alphas


if __name__ == "__main__":
    # ---------- Step 2: 合成資料驗證PCA ----------
    np.random.seed(42)  # 固定亂數種子,結果可重現
    n_samples = 500

    # 資料點分布在半徑 3 的圓周上(t 是角度)加一點雜訊;第三個特徵 x3 幾乎是 x1、x2 的線性組合,
    # 所以 3 維資料其實只有 2 個真正獨立的方向,PCA 應該能壓到 2 維而幾乎不損失資訊
    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

    # column_stack 把三個 (500,) 的一維陣列並排成 (500, 3) 的矩陣,每欄一個特徵
    X_synthetic = np.column_stack([x1, x2, x3])

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_synthetic)

    print(f"Original shape: {X_synthetic.shape}")  # (500, 3)
    print(f"Reduced shape:  {X_reduced.shape}")    # (500, 2)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

    # ---------- 還原誤差 demo ----------
    X_hat = pca.inverse_transform(X_reduced)  # 壓成 2 維再還原回 3 維
    err = reconstruction_error(X_synthetic, X_hat)
    print(f"\nReconstruction error (k=2 of 3 dims): {err:.6f}")

    # ---------- 跟sklearn對照 ----------
    # 在 __main__ 裡才 import sklearn,讓上面的 PCA class 本身不依賴它
    from sklearn.decomposition import PCA as SklearnPCA

    sklearn_pca = SklearnPCA(n_components=2)
    X_sklearn = sklearn_pca.fit_transform(X_synthetic)
    print(f"\nOur PCA explained variance:     {pca.explained_variance_ratio_}")
    print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")
    # 特徵向量的正負號是任意的(v 和 -v 都是合法答案),所以比較前先取絕對值,再看最大差距
    diff = np.abs(np.abs(X_reduced) - np.abs(X_sklearn))
    print(f"Max absolute difference: {diff.max():.10f}")

    # ---------- Kernel PCA demo:同心圓(標準PCA分不開,kernel PCA可以) ----------
    from sklearn.datasets import make_circles

    # make_circles 產生內外兩圈同心圓資料;factor=0.3 是內圈半徑為外圈的 0.3 倍
    X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)

    kpca = KernelPCA(n_components=2, gamma=10)
    X_kpca = kpca.fit_transform(X_circles)
    print(f"\nKernel PCA on concentric circles, output shape: {X_kpca.shape}")  # (300, 2)

    linear_pca = PCA(n_components=2)
    X_linear = linear_pca.fit_transform(X_circles)
    print(f"Standard PCA explained variance on circles: {linear_pca.explained_variance_ratio_}")
    # 標準PCA對同心圓幾乎沒用(兩個方向變異量差不多,因為圓形資料沒有明顯的線性主軸)

    # === 🟢 ===
    # t-SNE / UMAP demo,課程3個Exercises尚未做
