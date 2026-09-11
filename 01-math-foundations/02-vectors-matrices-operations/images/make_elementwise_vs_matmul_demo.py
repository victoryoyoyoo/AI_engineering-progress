"""
element-wise multiply vs matrix multiply對照圖:同樣兩個矩陣,兩種完全不同的運算方式
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
elem = A * B
mat = A @ B

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

def draw_grid(ax, M, title, color, highlight=None):
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(0, M.shape[0])
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=color)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            fc = "#FDECEC" if highlight == (i, j) else "#F0F0F0"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="#999999"))
            ax.text(j + 0.5, i + 0.5, str(M[i, j]), ha="center", va="center", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

draw_grid(axes[0], elem, "A * B (element-wise)\n同位置互乘,形狀不變", "#4C72B0")
axes[0].text(1, 2.3, "[0][0] = 1*5 = 5\n(對應位置直接乘)", fontsize=9, ha="center", color="#4C72B0")

draw_grid(axes[1], mat, "A @ B (matrix multiply)\n列跟行做內積", "#C44E52")
axes[1].text(1, 2.3, "[0][0] = 1*5+2*7 = 19\n(row0 · col0,做內積)", fontsize=9, ha="center", color="#C44E52")

fig.suptitle("Element-wise vs Matrix Multiply:同樣的A、B,結果完全不同")
plt.tight_layout()
plt.savefig("elementwise_vs_matmul.png", dpi=150, bbox_inches="tight")
print("saved")
