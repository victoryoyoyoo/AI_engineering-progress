import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, GOLD

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

setup_style()

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.2)
ax.axis("off")

ax.text(6.5, 4.95, "記憶體裡永遠是一長條:1 2 3 4 5 6(不管幾維)", ha="center", fontsize=12, fontweight="bold")
for i in range(6):
    ax.add_patch(Rectangle((3.2 + i * 1.0, 3.9), 1.0, 0.8, facecolor=GRAY, alpha=0.25, edgecolor=GRAY, linewidth=1.6))
    ax.text(3.7 + i * 1.0, 4.3, str(i + 1), ha="center", va="center", fontsize=13)
    ax.text(3.7 + i * 1.0, 3.7, f"位置{i}", ha="center", fontsize=8, color=GRAY)

def small_grid(x0, y0, rows, cols, vals, color):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * 0.7, y0 - r * 0.7), 0.7, 0.7, facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=1.5))
            ax.text(x0 + c * 0.7 + 0.35, y0 - r * 0.7 + 0.35, str(vals[r][c]), ha="center", va="center", fontsize=12)

ax.text(3.0, 2.95, "a  shape (2, 3)\nstrides (3, 1)", ha="center", fontsize=11, fontweight="bold", color=BLUE)
small_grid(1.9, 2.0, 2, 3, [[1, 2, 3], [4, 5, 6]], BLUE)
ax.text(3.0, 0.35, "往下一列 = 跳 3 格\n往右一欄 = 跳 1 格", ha="center", fontsize=10)

ax.annotate("", xy=(7.6, 1.6), xytext=(5.3, 1.6), arrowprops=dict(arrowstyle="->", lw=2.5, color=GRAY))
ax.text(6.45, 2.0, "a.T(轉置)", ha="center", fontsize=11)
ax.text(6.45, 1.05, "資料一個都沒動\n只把 strides 對調", ha="center", fontsize=10, color=RED)

ax.text(10.2, 2.95, "a.T  shape (3, 2)\nstrides (1, 3)", ha="center", fontsize=11, fontweight="bold", color=GREEN)
small_grid(9.5, 2.0, 3, 2, [[1, 4], [2, 5], [3, 6]], GREEN)
ax.text(10.2, 0.0, "往下一列 = 跳 1 格\n往右一欄 = 跳 3 格", ha="center", fontsize=10)

plt.savefig(os.path.join(os.path.dirname(__file__), "strides.png"), bbox_inches="tight")
print("saved")
