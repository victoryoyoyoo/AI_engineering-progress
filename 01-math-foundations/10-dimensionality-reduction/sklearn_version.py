"""
Lesson 10: 用scikit-learn對照
"""
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA, KernelPCA as SklearnKernelPCA
from sklearn.datasets import make_circles


def demo_sklearn_pca():
    np.random.seed(42)
    n_samples = 500
    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)
    X = np.column_stack([x1, x2, x3])

    pca = SklearnPCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print("Sklearn PCA explained variance ratios:", pca.explained_variance_ratio_)
    print("Sklearn PCA components shape:", pca.components_.shape)
    return X_reduced


def demo_sklearn_kernel_pca():
    X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    kpca = SklearnKernelPCA(n_components=2, kernel="rbf", gamma=10)
    X_kpca = kpca.fit_transform(X_circles)
    print("Sklearn KernelPCA output shape:", X_kpca.shape)
    return X_kpca


if __name__ == "__main__":
    demo_sklearn_pca()
    demo_sklearn_kernel_pca()
