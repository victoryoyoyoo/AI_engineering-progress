import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

setup_style()

A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
R = A + b

fig, ax = plt.subplots(figsize=(13, 4.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.4)
ax.axis("off")


def draw(x0, y0, mat, color, alpha=0.3, dashed=False, cell=0.75):
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                   facecolor=color, alpha=alpha, edgecolor=color,
                                   linewidth=1.6, linestyle="--" if dashed else "-"))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, str(mat[r, c]),
                    ha="center", va="center", fontsize=12)


ax.text(1.4, 4.0, "A  shape (2, 3)", ha="center", fontsize=11, fontweight="bold")
draw(0.2, 2.6, A, BLUE)

ax.text(2.95, 2.7, "+", fontsize=22, ha="center")

ax.text(5.1, 4.0, "b  shape (3,)", ha="center", fontsize=11, fontweight="bold")
draw(3.9, 2.6, b.reshape(1, 3), RED)
ax.text(5.1, 1.75, "只有一排,但 A 有兩排\n→ 自動把這排複製一份", ha="center", fontsize=9.5)
draw(3.9, 1.1, b.reshape(1, 3), RED, alpha=0.12, dashed=True)

ax.text(6.9, 2.7, "=", fontsize=22, ha="center")

ax.text(8.9, 4.0, "拉大後實際相加", ha="center", fontsize=11, fontweight="bold")
draw(7.8, 3.05, A, BLUE, alpha=0.3, cell=0.62)
ax.text(9.85, 2.4, "+", fontsize=16, ha="center")
draw(7.8, 1.7, np.array([[10, 20, 30], [10, 20, 30]]), RED, alpha=0.2, dashed=True, cell=0.62)

ax.text(11.5, 2.7, "=", fontsize=22, ha="center")
ax.text(12.0, 4.0, "結果 (2, 3)", ha="center", fontsize=11, fontweight="bold")
draw(10.55, 2.6, R, GREEN, cell=0.62)

plt.savefig(os.path.join(os.path.dirname(__file__), "broadcast_bias.png"), bbox_inches="tight")
print("saved")
