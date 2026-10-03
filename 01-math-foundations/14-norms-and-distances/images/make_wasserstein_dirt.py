import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, DPI

import matplotlib.pyplot as plt

setup_style()

P = [0.5, 0.5, 0.0, 0.0]
Q = [0.0, 0.0, 0.5, 0.5]
pos = [0, 1, 2, 3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)

for ax, vals, color, name in [(ax1, P, BLUE, "P 的土堆\n(土的量)"), (ax2, Q, RED, "Q 的土堆\n(土的量)")]:
    ax.bar(pos, vals, color=color, width=0.5, alpha=0.85)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(name, fontsize=11.5, fontweight="bold", color=color)
    ax.set_xticks(pos)
    ax.grid(axis="y", color="#e3e3e3")

# 在兩張圖之間畫出搬運方向
for src, dst in [(0, 2), (1, 3)]:
    fig.add_artist(plt.matplotlib.patches.ConnectionPatch(
        xyA=(src, 0.0), coordsA=ax1.transData, xyB=(dst, 0.5), coordsB=ax2.transData,
        arrowstyle="-|>", color=GREEN, linewidth=2.6, mutation_scale=18))

ax1.text(0.9, 0.62, "每堆搬 0.5,距離 2\n工 = 0.5 × 2 = 1", color=GREEN, fontsize=10.5, fontweight="bold")
ax2.set_xlabel("位置")
fig.suptitle("搬土距離:把 P 的土堆搬成 Q,總共要花 1 + 1 = 2", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(os.path.dirname(__file__), "wasserstein_dirt.png"), dpi=DPI)
