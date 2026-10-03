import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GRAY, DPI

import numpy as np
import matplotlib.pyplot as plt

setup_style()

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

# 左圖:從 A(1,1) 走到 B(4,5),三種距離各自「怎麼量」
ax = axes[0]
ax.set_title("同樣從 A 到 B,三種量法")
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 6.2)
ax.set_aspect("equal")
ax.set_xticks(range(0, 6))
ax.set_yticks(range(0, 7))
ax.grid(True, color="#dddddd", linewidth=0.8)
A, B = (1, 1), (4, 5)
# L1:沿著街道走,先東3格再北4格
ax.plot([1, 4, 4], [1, 1, 5], color=BLUE, linewidth=3, label="L1 = 3 + 4 = 7(沿街走)")
# L2:直線
ax.plot([1, 4], [1, 5], color=RED, linewidth=3, label="L2 = √(9+16) = 5(直線)")
# L-inf:只看差最多的那一格
ax.annotate("", xy=(1, 5), xytext=(1, 1), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.5))
ax.text(0.15, 3.0, "L-inf = 4\n(只看最大的\n那個差)", color=GREEN, fontsize=9.5, fontweight="bold")
ax.scatter(*A, s=90, color="black", zorder=5)
ax.scatter(*B, s=90, color="black", zorder=5)
ax.text(0.8, 0.55, "A(1,1)", fontsize=10)
ax.text(4.1, 5.1, "B(4,5)", fontsize=10)
ax.legend(loc="lower right")

# 右圖:「離原點距離 = 1」的所有點,三種範數畫出來形狀不同
ax = axes[1]
ax.set_title("距離原點剛好 = 1 的所有點(單位球)")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.axhline(0, color="#888888", linewidth=0.8)
ax.axvline(0, color="#888888", linewidth=0.8)
t = np.linspace(0, 2 * np.pi, 400)
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=BLUE, linewidth=3, label="L1:菱形(角落在軸上)")
ax.plot(np.cos(t), np.sin(t), color=RED, linewidth=3, label="L2:圓形")
ax.plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], color=GREEN, linewidth=3, label="L-inf:正方形")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02))

fig.suptitle("L1 / L2 / L-inf:同一組點,量法不同,「遠近」就不同", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(os.path.dirname(__file__), "norm_paths_balls.png"), dpi=DPI)
