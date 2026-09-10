"""
Kernel PCA示意圖:同心圓資料,標準PCA分不開,Kernel PCA可以
"""
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "..")
from reference import PCA, KernelPCA
from sklearn.datasets import make_circles

X_circles, y_circles = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)

pca = PCA(n_components=2)
X_linear = pca.fit_transform(X_circles)

kpca = KernelPCA(n_components=2, gamma=10)
X_kpca = kpca.fit_transform(X_circles)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap="coolwarm", s=15)
axes[0].set_title("原始資料(同心圓)", fontsize=12, fontweight="bold")
axes[0].set_aspect("equal")

axes[1].scatter(X_linear[:, 0], X_linear[:, 1], c=y_circles, cmap="coolwarm", s=15)
axes[1].set_title(f"標準PCA\n(explained var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})\n兩群混在一起,分不開", fontsize=11, color="#C44E52")

axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap="coolwarm", s=15)
axes[2].set_title("Kernel PCA (RBF, gamma=10)\n兩群被拆開了", fontsize=12, color="#55A868", fontweight="bold")

plt.tight_layout()
plt.savefig("kernel_pca_circles.png", dpi=150, bbox_inches="tight")
print("saved")
