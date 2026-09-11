"""
determinant示意圖:矩陣把單位正方形變成平行四邊形,面積比例就是determinant;
det=0代表被壓扁成一條線,資訊遺失
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

matrices = {
    "A = [[2,0],[0,1.5]]\ndet=3.0 (面積放大3倍)": np.array([[2, 0], [0, 1.5]]),
    "B = [[1,0.6],[0.3,1]]\ndet=0.82 (輕微形變)": np.array([[1, 0.6], [0.3, 1]]),
    "C = [[2,1],[4,2]]\ndet=0 (壓扁成一條線)": np.array([[2, 1], [4, 2]]),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52"]

for ax, (title, M), c in zip(axes, matrices.items(), colors):
    transformed = unit_square @ M.T
    ax.plot(unit_square[:, 0], unit_square[:, 1], "--", color="#999999", lw=1.3, label="原始單位正方形")
    ax.fill(unit_square[:, 0], unit_square[:, 1], color="#999999", alpha=0.12)
    ax.plot(transformed[:, 0], transformed[:, 1], color=c, lw=2.4, label="轉換後")
    ax.fill(transformed[:, 0], transformed[:, 1], color=c, alpha=0.28)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=c)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.axvline(0, color="#bbbbbb", lw=0.6)
    ax.legend(loc="upper left")

fig.suptitle("Determinant:單位正方形被放大/縮小/壓扁的比例")
plt.tight_layout()
plt.savefig("determinant_area_scaling.png", dpi=150, bbox_inches="tight")
print("saved")
