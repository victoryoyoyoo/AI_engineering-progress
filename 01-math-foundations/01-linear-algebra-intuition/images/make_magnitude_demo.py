"""
magnitude(向量長度)示意圖:畢氏定理的推廣,向量長度是各分量平方和開根號
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

v = np.array([4, 3])
mag = float(np.linalg.norm(v))  # sqrt(4^2+3^2) = 5

fig, ax = plt.subplots(figsize=(6, 6))
# 向量本身(斜邊)
ax.annotate("", xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#4C72B0", lw=2.8, mutation_scale=22))
# 直角三角形的兩條邊,對應向量的兩個分量
ax.plot([0, v[0]], [0, 0], color="#55A868", lw=2.2, linestyle="--")
ax.plot([v[0], v[0]], [0, v[1]], color="#C44E52", lw=2.2, linestyle="--")

ax.text(v[0] / 2, -0.35, "x = 4", color="#55A868", fontsize=11, ha="center", fontweight="bold")
ax.text(v[0] + 0.25, v[1] / 2, "y = 3", color="#C44E52", fontsize=11, va="center", fontweight="bold")
ax.text(v[0] / 2 - 0.3, v[1] / 2 + 0.3, f"|v| = sqrt(4²+3²) = {mag:.0f}", color="#4C72B0",
        fontsize=11, fontweight="bold", rotation=37)

ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlim(-1, 5.5)
ax.set_ylim(-1, 4.5)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("magnitude(向量長度):畢氏定理的推廣")

plt.tight_layout()
plt.savefig("magnitude.png", dpi=150, bbox_inches="tight")
print("saved")
