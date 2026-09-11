"""
transpose(轉置)示意圖:矩陣的行跟列互換,[i][j] 轉置後變成 [j][i]
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

A = np.array([[1, 2, 3], [4, 5, 6]])
AT = A.T

fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))

for ax, M, title, hi in zip(axes, [A, AT], [f"A  shape={A.shape}", f"A.T  shape={AT.shape}"], [(0, 2), (2, 0)]):
    ax.imshow(np.zeros_like(M, dtype=float), cmap="Greys", vmin=0, vmax=1, alpha=0.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "#C44E52" if (i, j) == hi else "#4C72B0"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, alpha=0.18 if color == "#4C72B0" else 0.35, edgecolor="black"))
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(M.shape[0] - 0.5, -0.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

axes[0].text(2, 0, "", fontsize=1)
fig.suptitle("transpose(轉置):A[0][2]=3 轉置後變成 A.T[2][0]=3(紅框標示同一個數字)")
plt.tight_layout()
plt.savefig("transpose.png", dpi=150, bbox_inches="tight")
print("saved")
