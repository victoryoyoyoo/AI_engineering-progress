import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, DPI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

setup_style()

# 跟 reference.py 的 demo_mahalanobis 同一份資料
random.seed(42)
data = []
for _ in range(200):
    x = random.gauss(0, 3)
    y = 0.8 * x + random.gauss(0, 1)
    data.append([x, y])
data = np.array(data)
mean = data.mean(axis=0)
cov = np.cov(data, rowvar=False)

along = mean + np.array([3, 0.8 * 3])
perp = mean + np.array([1, -3])

fig, ax = plt.subplots(figsize=(8.4, 6.6))
ax.scatter(data[:, 0], data[:, 1], s=14, color=GRAY, alpha=0.45, label="資料(兩個特徵會一起變大變小)")
ax.scatter(*mean, s=90, color="black", zorder=5)
ax.text(mean[0] + 0.25, mean[1] - 0.95, "平均", fontsize=10.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))

# 直線距離:圓
ax.add_patch(Circle(mean, 3.5, fill=False, color=BLUE, linestyle="--", linewidth=2))
# Mahalanobis 距離 = 2 的等高線:順著資料拉長的橢圓
vals, vecs = np.linalg.eigh(cov)
angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
ax.add_patch(Ellipse(mean, 2 * 2 * math.sqrt(vals[1]), 2 * 2 * math.sqrt(vals[0]), angle=angle,
                     fill=False, color=RED, linewidth=2.4))

ax.scatter(*along, s=110, color=GREEN, zorder=6)
ax.scatter(*perp, s=110, color=RED, zorder=6)
ax.annotate("點1(順著資料)\n直線 3.84,Mahalanobis 0.98 → 正常", xy=along, xytext=(-8.6, 6.4), color=GREEN, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
ax.annotate("點2(橫過資料)\n直線 3.16,Mahalanobis 3.86 → 異常", xy=perp, xytext=(-1.2, -6.9), color=RED, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))

ax.set_xlim(-9, 9)
ax.set_ylim(-8, 8)
ax.set_aspect("equal")
ax.grid(True, color="#e3e3e3")
ax.set_title("藍色虛線圓:直線距離相同的點\n紅色橢圓:Mahalanobis 距離相同的點")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "mahalanobis_ellipse.png"), dpi=DPI)
