import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components + self.mean


def reconstruction_error(X, X_reconstructed):
    return np.mean((X - X_reconstructed) ** 2)


class KernelPCA:
    def __init__(self, n_components, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.X_fit = None
        self.alphas = None
        self.lambdas = None

    def _rbf_kernel(self, X1, X2):
        sq_dists = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        return np.exp(-self.gamma * sq_dists)

    def fit_transform(self, X):
        self.X_fit = X
        n = X.shape[0]

        K = self._rbf_kernel(X, X)

        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.lambdas = eigenvalues[:self.n_components]
        self.alphas = eigenvectors[:, :self.n_components] / np.sqrt(
            np.maximum(self.lambdas, 1e-12)
        )

        return K_centered @ self.alphas


def curse_of_dimensionality_demo():
    np.random.seed(0)
    dims = [1, 2, 3, 5, 10, 50, 100, 500, 1000]
    ratios = []
    n_points = 1000

    for d in dims:
        X = np.random.uniform(0, 1, size=(n_points, d))
        origin = np.zeros(d)
        dists = np.linalg.norm(X - origin, axis=1)
        ratio = (dists.max() - dists.min()) / dists.min()
        ratios.append(ratio)

    return dims, ratios


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 500
    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
    x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
    x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)
    X_synthetic = np.column_stack([x1, x2, x3])

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_synthetic)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

    X_hat = pca.inverse_transform(X_reduced)
    err = reconstruction_error(X_synthetic, X_hat)
    print(f"Reconstruction error: {err:.6f}")

    from sklearn.datasets import make_circles

    X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    kpca = KernelPCA(n_components=2, gamma=10)
    X_kpca = kpca.fit_transform(X_circles)
    print(f"Kernel PCA output shape: {X_kpca.shape}")

    linear_pca = PCA(n_components=2)
    X_linear = linear_pca.fit_transform(X_circles)
    print(f"Standard PCA explained variance on circles: {linear_pca.explained_variance_ratio_}")

    dims, ratios = curse_of_dimensionality_demo()
    for d, r in zip(dims, ratios):
        print(f"dim={d:5d}  ratio={r:.3f}")
