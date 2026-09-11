"""
PCA示意圖:協方差矩陣最大eigenvalue對應的eigenvector,就是資料變異量最大的方向(第一主成分)
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

rng = np.random.default_rng(7)
transform = np.array([[2.2, 1.1], [0.3, 0.6]])
raw = rng.normal(size=(300, 2))
data = raw @ transform.T

cov = np.cov(data.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(data[:, 0], data[:, 1], s=10, color="#4C72B0", alpha=0.35, label="資料點")

colors = ["#C44E52", "#55A868"]
labels = ["第1主成分(最大eigenvalue,變異量最大方向)", "第2主成分(次大eigenvalue)"]
for i, c, lab in zip(range(2), colors, labels):
    v = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2.2
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=3, mutation_scale=22))
    ax.annotate("", xy=-v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=c, lw=3))
    ax.plot([], [], color=c, lw=3, label=lab)

ax.axhline(0, color="#bbbbbb", lw=0.6)
ax.axvline(0, color="#bbbbbb", lw=0.6)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="upper left")
ax.set_title("PCA:協方差矩陣的eigenvector就是主成分方向")

plt.tight_layout()
plt.savefig("pca_direction.png", dpi=150, bbox_inches="tight")
print("saved")
