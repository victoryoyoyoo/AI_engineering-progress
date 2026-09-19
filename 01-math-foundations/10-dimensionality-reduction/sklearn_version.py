"""
Lesson 10: 用scikit-learn對照
reference.py 手刻的 PCA、KernelPCA,在 scikit-learn 裡各是一個現成的 class,
用法遵循 sklearn 統一的 fit / transform / fit_transform 介面。
"""
import numpy as np
# 改名 import:避免與 reference.py 手刻的同名 class 混淆
from sklearn.decomposition import PCA as SklearnPCA, KernelPCA as SklearnKernelPCA
from sklearn.datasets import make_circles  # 產生同心圓測試資料


def demo_sklearn_pca():
    np.random.seed(42)
    n_samples = 500
    # 資料產生方式與 reference.py 完全相同:圓周上的點 + 一個幾乎是線性組合的第三特徵
    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)
    X = np.column_stack([x1, x2, x3])  # shape (500, 3)

    pca = SklearnPCA(n_components=2)     # 降到 2 維
    # fit_transform = fit(算出主成分方向)+ transform(投影),一步完成,對應手刻版的同名方法
    X_reduced = pca.fit_transform(X)
    # explained_variance_ratio_:每個主成分佔總變異量的比例,結尾底線是 sklearn「訓練後才有」的命名慣例
    print("Sklearn PCA explained variance ratios:", pca.explained_variance_ratio_)
    # components_ 的 shape 是 (2, 3):2 個主成分方向,每個方向是 3 維向量,對應手刻版的 self.components
    print("Sklearn PCA components shape:", pca.components_.shape)
    return X_reduced


def demo_sklearn_kernel_pca():
    # 同心圓:標準 PCA(線性)分不開,Kernel PCA 可以
    X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    # kernel="rbf" 指定用 RBF 核函數;gamma 越大,只有很近的點才算相似
    kpca = SklearnKernelPCA(n_components=2, kernel="rbf", gamma=10)
    X_kpca = kpca.fit_transform(X_circles)
    print("Sklearn KernelPCA output shape:", X_kpca.shape)  # (300, 2)
    return X_kpca


if __name__ == "__main__":
    demo_sklearn_pca()
    demo_sklearn_kernel_pca()
