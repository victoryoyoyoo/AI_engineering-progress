"""
特徵向量/特徵值示意圖:矩陣變換下,方向不變、只被拉伸的特殊方向
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

M = np.array([[2.0, 0.5], [0.5, 1.5]])
eigenvalues, eigenvectors = np.linalg.eigh(M)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

colors = ["#4C72B0", "#C44E52", "#55A868"]
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.column_stack([np.cos(theta), np.sin(theta)])

axes[0].plot(circle[:, 0], circle[:, 1], "--", color="gray", alpha=0.5)
sample_vecs = [np.array([1, 0]), np.array([0, 1]), np.array([0.7, 0.7]), np.array([-0.7, 0.7])]
for v, c in zip(sample_vecs, colors + ["#8172B2"]):
    axes[0].arrow(0, 0, v[0], v[1], head_width=0.06, color=c, alpha=0.6, length_includes_head=True)
axes[0].set_title("變換前:一般向量", fontsize=12, fontweight="bold")
axes[0].set_xlim(-1.5, 1.5)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_aspect("equal")
axes[0].grid(alpha=0.3)

ellipse = circle @ M.T
axes[1].plot(ellipse[:, 0], ellipse[:, 1], "--", color="gray", alpha=0.5)
for v, c in zip(sample_vecs, colors + ["#8172B2"]):
    v2 = M @ v
    axes[1].arrow(0, 0, v2[0], v2[1], head_width=0.09, color=c, alpha=0.6, length_includes_head=True)

for i in range(2):
    ev = eigenvectors[:, i] * eigenvalues[i]
    axes[1].arrow(0, 0, ev[0], ev[1], head_width=0.09, color="black", linewidth=2.5,
                  length_includes_head=True, zorder=5)
    axes[1].text(ev[0] * 1.15, ev[1] * 1.15, f"特徵向量{i+1}\n(特徵值={eigenvalues[i]:.2f})",
                 fontsize=9, fontweight="bold")

axes[1].set_title("變換後(乘上矩陣M):黑色=特徵向量方向不變,只被拉伸", fontsize=11, fontweight="bold")
axes[1].set_xlim(-2.5, 2.5)
axes[1].set_ylim(-2.5, 2.5)
axes[1].set_aspect("equal")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eigenvector_demo.png", dpi=150, bbox_inches="tight")
print("saved")
