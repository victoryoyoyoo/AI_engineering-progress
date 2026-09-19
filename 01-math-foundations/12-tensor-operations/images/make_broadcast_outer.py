import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

setup_style()

a = np.array([[1], [2], [3]])
b = np.array([[10, 20, 30, 40]])
A_big = np.repeat(a, 4, axis=1)
B_big = np.repeat(b, 3, axis=0)
R = a * b

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.6)
ax.axis("off")


def draw(x0, y0, mat, color, alpha=0.3, dashed=False, cell=0.6, real=None):
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            is_real = real is None or real(r, c)
            ax.add_patch(Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                   facecolor=color, alpha=alpha if is_real else 0.1,
                                   edgecolor=color, linewidth=1.5,
                                   linestyle="-" if is_real else "--"))
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, str(mat[r, c]),
                    ha="center", va="center", fontsize=11, alpha=1.0 if is_real else 0.5)


ax.text(1.3, 4.25, "a  shape (3, 1)", ha="center", fontsize=11, fontweight="bold")
draw(0.6, 3.5, A_big, BLUE, real=lambda r, c: c == 0)
ax.text(1.55, 1.55, "只有 1 欄\n→ 往右複製成 4 欄", ha="center", fontsize=9.5)

ax.text(3.9, 3.0, "×", fontsize=22, ha="center")

ax.text(6.2, 4.25, "b  shape (1, 4)", ha="center", fontsize=11, fontweight="bold")
draw(5.0, 3.5, B_big, RED, real=lambda r, c: r == 0)
ax.text(6.2, 1.55, "只有 1 列\n→ 往下複製成 3 列", ha="center", fontsize=9.5)

ax.text(7.9, 3.0, "=", fontsize=22, ha="center")

ax.text(10.4, 4.25, "結果 shape (3, 4)", ha="center", fontsize=11, fontweight="bold")
draw(9.1, 3.5, R, GREEN)

plt.savefig(os.path.join(os.path.dirname(__file__), "broadcast_outer.png"), bbox_inches="tight")
print("saved")
