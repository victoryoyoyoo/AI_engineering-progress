"""
還原誤差示意圖:原始點 vs PCA降維再還原後的點,誤差就是兩者的距離
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
from reference import PCA, reconstruction_error

np.random.seed(42)
n_samples = 60
t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
X = np.column_stack([x1, x2])

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
X_hat = pca.inverse_transform(X_reduced)
err = reconstruction_error(X, X_hat)

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(X[:, 0], X[:, 1], s=30, color="#4C72B0", label="原始資料", zorder=3)
ax.scatter(X_hat[:, 0], X_hat[:, 1], s=30, color="#C44E52", marker="x", label="還原後(降到1維再還原)", zorder=3)
for i in range(n_samples):
    ax.plot([X[i, 0], X_hat[i, 0]], [X[i, 1], X_hat[i, 1]], color="gray", alpha=0.4, linewidth=0.8)

ax.set_title(f"還原誤差示意:灰線=每筆資料的還原誤差\nMSE = {err:.4f}", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reconstruction_error_demo.png", dpi=150, bbox_inches="tight")
print("saved")
