"""
凸函數 vs 非凸函數示意圖:凸函數只有一個最小值,非凸函數有局部最小值/鞍點/平坦區域
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

x = np.linspace(-3, 3, 300)
convex = x ** 2
nonconvex = 0.15 * x ** 4 - 0.9 * x ** 2 + 0.3 * x + 3

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

axes[0].plot(x, convex, color="#4C72B0", lw=2.5)
axes[0].plot(0, 0, "*", color="#55A868", markersize=16, zorder=5)
axes[0].set_title("凸函數 f(x)=x²\n只有一個最小值,梯度下降一定找得到", fontsize=11.5, fontweight="bold", color="#4C72B0")

axes[1].plot(x, nonconvex, color="#C44E52", lw=2.5)
local_min_x = -1.7
global_min_x = 1.85
axes[1].plot(local_min_x, 0.15 * local_min_x ** 4 - 0.9 * local_min_x ** 2 + 0.3 * local_min_x + 3,
             "o", color="#8172B2", markersize=11, zorder=5, label="局部最小值")
axes[1].plot(global_min_x, 0.15 * global_min_x ** 4 - 0.9 * global_min_x ** 2 + 0.3 * global_min_x + 3,
             "*", color="#55A868", markersize=16, zorder=5, label="全域最小值")
axes[1].legend()
axes[1].set_title("非凸函數(神經網路loss長這樣)\n有局部最小值、鞍點、平坦區域", fontsize=11.5, fontweight="bold", color="#C44E52")

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3, linestyle="--")

fig.suptitle("Convex vs Non-convex:神經網路的loss地形比凸函數複雜很多")
plt.tight_layout()
plt.savefig("convex_vs_nonconvex.png", dpi=150, bbox_inches="tight")
print("saved")
