"""
linear independence(線性獨立) 與 rank(秩) 示意圖:
左邊三個向量互相獨立(撐出滿滿的2維空間) vs 右邊第三個向量是前兩個的線性組合(冗餘,不增加維度)
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

# 左圖:兩個線性獨立的向量,rank = 2
v1, v2 = np.array([2, 0.5]), np.array([0.5, 2])
ax = axes[0]
for v, c, lab in zip([v1, v2], ["#4C72B0", "#55A868"], ["v1", "v2"]):
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, mutation_scale=22))
    ax.text(v[0] + 0.1, v[1] + 0.1, lab, color=c, fontsize=12, fontweight="bold")
ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlim(-1, 3); ax.set_ylim(-1, 3); ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("線性獨立:v1、v2 撐出整個2維平面\nrank = 2(沒有冗餘方向)", fontsize=11.5, fontweight="bold")

# 右圖:v3 是 v1、v2 的線性組合,加了也不增加維度,rank 仍是 2
v3 = 1.2 * v1 + 0.4 * v2
ax = axes[1]
for v, c, lab in zip([v1, v2, v3], ["#4C72B0", "#55A868", "#C44E52"], ["v1", "v2", "v3 = 1.2v1+0.4v2"]):
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, mutation_scale=22,
                                 linestyle="--" if lab.startswith("v3") else "-"))
    ax.text(v[0] + 0.1, v[1] + 0.1, lab, color=c, fontsize=10, fontweight="bold")
ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlim(-1, 4); ax.set_ylim(-1, 4); ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("線性相依:v3 可由 v1、v2 組合出來\n加入 v3 仍是 rank = 2(v3 是冗餘方向)", fontsize=11.5, fontweight="bold")

fig.suptitle("linear independence(線性獨立) 與 rank(矩陣的秩)", y=1.02)
plt.tight_layout()
plt.savefig("rank_independence.png", dpi=150, bbox_inches="tight")
print("saved")
