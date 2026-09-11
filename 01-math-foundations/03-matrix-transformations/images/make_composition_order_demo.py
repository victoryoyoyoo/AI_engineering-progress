"""
複合變換順序示意圖:B@A@點 跟 A@B@點 是不一樣的結果,矩陣乘法沒有交換律
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

rot90 = np.array([[0, -1], [1, 0]])
scale = np.array([[1, 0], [0, 0.5]])
p = np.array([1, 1])

# 先轉再縮放: scale @ rot90 @ p
mid1 = rot90 @ p
final1 = scale @ mid1
# 先縮放再轉: rot90 @ scale @ p
mid2 = scale @ p
final2 = rot90 @ mid2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

def draw_path(ax, pts, labels, colors, title, color):
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.axvline(0, color="#bbbbbb", lw=0.6)
    for i in range(len(pts) - 1):
        ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.2, mutation_scale=18))
    for pt, lab, c in zip(pts, labels, colors + [colors[-1]]):
        ax.plot(*pt, "o", color=c, markersize=7)
        ax.annotate(lab, pt, textcoords="offset points", xytext=(8, 6), fontsize=10, color=c)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-0.3, 1.7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=color)

draw_path(axes[0], [p, mid1, final1], ["p=(1,1)", "先轉90°", "再縮放y*0.5"],
          ["#4C72B0", "#55A868"], "scale @ rot90 @ p\n先轉再縮放", "#4C72B0")
draw_path(axes[1], [p, mid2, final2], ["p=(1,1)", "先縮放y*0.5", "再轉90°"],
          ["#C44E52", "#8172B2"], "rot90 @ scale @ p\n先縮放再轉", "#C44E52")

f1 = tuple(float(v) for v in np.round(final1, 2))
f2 = tuple(float(v) for v in np.round(final2, 2))
fig.suptitle(f"複合變換順序影響結果:{f1} vs {f2}")
plt.tight_layout()
plt.savefig("composition_order_matters.png", dpi=150, bbox_inches="tight")
print("saved")
