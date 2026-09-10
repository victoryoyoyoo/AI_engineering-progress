"""
PCA / t-SNE / UMAP 三種降維法對照:同一份多群資料,各自降到2維後的樣子
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE
import umap

X, y = make_blobs(n_samples=600, n_features=20, centers=5, cluster_std=3.0, random_state=42)

pca = SklearnPCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
cmap = "tab10"

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, s=12, alpha=0.7)
axes[0].set_title("PCA\n(線性,保留全域變異方向)", fontsize=12, fontweight="bold")

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, s=12, alpha=0.7)
axes[1].set_title("t-SNE\n(保留鄰居關係,群更緊實分開)", fontsize=12, fontweight="bold")

axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=cmap, s=12, alpha=0.7)
axes[2].set_title("UMAP\n(類似t-SNE但更快、更保留全域結構)", fontsize=12, fontweight="bold")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("同一份20維、5群的資料,降到2維:三種方法的取捨", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pca_tsne_umap_comparison.png", dpi=150, bbox_inches="tight")
print("saved")
