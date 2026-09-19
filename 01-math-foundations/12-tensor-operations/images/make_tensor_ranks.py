import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

setup_style()

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.2)
ax.axis("off")


def grid(x0, y0, rows, cols, color, cell=0.42):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                   facecolor=color, alpha=0.55, edgecolor="white", linewidth=1.5))


ax.text(1.0, 3.9, "純量 Scalar", ha="center", fontsize=11, fontweight="bold")
grid(0.8, 2.6, 1, 1, GRAY)
ax.text(1.0, 1.7, "維度 = 0\nshape = ()", ha="center", fontsize=10)

ax.text(3.6, 3.9, "向量 Vector", ha="center", fontsize=11, fontweight="bold")
grid(2.7, 2.6, 1, 4, BLUE)
ax.text(3.6, 1.7, "維度 = 1\nshape = (4,)", ha="center", fontsize=10)

ax.text(6.9, 3.9, "矩陣 Matrix", ha="center", fontsize=11, fontweight="bold")
grid(6.0, 3.0, 3, 4, GREEN)
ax.text(6.9, 1.2, "維度 = 2\nshape = (3, 4)", ha="center", fontsize=10)

ax.text(11.0, 3.9, "3維 Tensor", ha="center", fontsize=11, fontweight="bold")
for k in range(3, -1, -1):
    grid(9.8 + k * 0.28, 3.0 - k * 0.28, 3, 3, PURPLE if k == 0 else GOLD, cell=0.36)
ax.text(11.0, 1.2, "維度 = 3\nshape = (4, 3, 3)\n= 4 張 3×3 疊起來", ha="center", fontsize=10)

fig.suptitle("Tensor 就是「維度不限的陣列」:維度 = 有幾個座標軸(axis)", fontsize=13, fontweight="bold")
plt.savefig(os.path.join(os.path.dirname(__file__), "tensor_ranks.png"), bbox_inches="tight")
print("saved")
