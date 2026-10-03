import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, DPI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

setup_style()

fig, ax = plt.subplots(figsize=(8.2, 6.2))
ax.set_xlim(-0.3, 4.6)
ax.set_ylim(-0.3, 4.6)
ax.set_aspect("equal")
ax.grid(True, color="#e3e3e3", linewidth=0.8)
ax.axhline(0, color="#888888", linewidth=0.8)
ax.axvline(0, color="#888888", linewidth=0.8)

A = (1, 2)
B = (2, 4)
C = (3, 1)


def arrow(p, color, label, dx, dy):
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=3))
    ax.text(p[0] + dx, p[1] + dy, label, color=color, fontsize=11.5, fontweight="bold")


arrow(A, BLUE, "A(1,2)", -0.75, 0.12)
arrow(B, GREEN, "B(2,4)  = A 放大 2 倍", 0.1, 0.05)
arrow(C, RED, "C(3,1)", 0.1, -0.05)

# A 和 B 的直線距離(L2):看起來很遠
ax.plot([A[0], B[0]], [A[1], B[1]], color=GRAY, linestyle="--", linewidth=1.6)
ax.text(1.62, 2.7, "A 到 B 的直線距離 ≈ 2.24\n(算很遠)", color=GRAY, fontsize=9.5)

# A 和 C 的夾角 45 度
ax.add_patch(Arc((0, 0), 1.9, 1.9, theta1=18.43, theta2=63.43, color="black", linewidth=1.8))
ax.text(1.05, 0.95, "45°", fontsize=11.5)

ax.set_title("A 和 B 方向完全一樣(夾角 0°)  →  方向相似度 = 1\nA 和 C 夾角 45°  →  方向相似度 ≈ 0.71")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "cosine_direction.png"), dpi=DPI)
