import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

setup_style()

img = np.array([[[1, 10, 100], [2, 20, 200]],
                [[3, 30, 300], [4, 40, 400]]])
colors = [RED, GREEN, BLUE]

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.6)
ax.axis("off")

ax.text(2.6, 4.25, "轉之前 (H, W, C)\n每個像素自己帶一組 [紅, 綠, 藍]", ha="center", fontsize=11, fontweight="bold")
for r in range(2):
    for c in range(2):
        x0, y0 = 0.6 + c * 2.0, 2.2 - r * 1.5
        ax.add_patch(Rectangle((x0, y0), 1.9, 1.3, facecolor="none", edgecolor=GRAY, linewidth=2))
        for k in range(3):
            ax.text(x0 + 0.32 + k * 0.62, y0 + 0.65, str(img[r, c, k]), color=colors[k],
                    ha="center", va="center", fontsize=12, fontweight="bold")

ax.annotate("", xy=(7.0, 2.4), xytext=(5.0, 2.4),
            arrowprops=dict(arrowstyle="->", lw=2.5, color=GRAY))
ax.text(6.0, 2.75, "transpose\n(0, 3, 1, 2)", ha="center", fontsize=10)
ax.text(6.0, 1.8, "數字沒變\n只是重新分組", ha="center", fontsize=10)

ax.text(10.4, 4.25, "轉之後 (C, H, W)\n每個顏色自己一張圖", ha="center", fontsize=11, fontweight="bold")
names = ["紅", "綠", "藍"]
for k in range(3):
    x0 = 7.3 + k * 2.0
    ax.text(x0 + 0.8, 3.75, names[k], color=colors[k], ha="center", fontsize=11, fontweight="bold")
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((x0 + c * 0.8 - 0.0, 2.6 - r * 0.9 + 0.0), 0.8, 0.9,
                                   facecolor=colors[k], alpha=0.25, edgecolor=colors[k], linewidth=1.5))
            ax.text(x0 + c * 0.8 + 0.4, 2.6 - r * 0.9 + 0.45, str(img[r, c, k]),
                    ha="center", va="center", fontsize=11)

plt.savefig(os.path.join(os.path.dirname(__file__), "transpose_channels.png"), bbox_inches="tight")
print("saved")
