import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt

setup_style()

A = np.array([[2.0, 1.0], [1.0, 3.0]])
U, S, Vt = np.linalg.svd(A)

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(theta), np.sin(theta)])

after_vt = Vt @ circle
after_sigma = np.diag(S) @ after_vt
after_u = U @ after_sigma

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

titles = [
    "1. 輸入空間:單位圓\n(還沒套用任何變換)",
    "2. 套用 Vᵀ(旋轉)\n方向改變,長度不變",
    "3. 再套用 Σ(縮放)\n沿軸拉伸,還沒轉到輸出空間",
]
data = [circle, after_vt, after_sigma]
colors = [GRAY, BLUE, RED]

for ax, title, pts, color in zip(axes, titles, data, colors):
    ax.plot(pts[0], pts[1], color=color, linewidth=2)
    ax.axhline(0, color=GRAY, linewidth=0.5)
    ax.axvline(0, color=GRAY, linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5)
    lim = max(np.abs(pts).max() * 1.3, 1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

fig.suptitle("SVD幾何意義:A = U × Σ × Vᵀ 做的事是「旋轉 → 縮放 → 旋轉」", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "rotate_scale_rotate.png"), bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(circle[0], circle[1], color=GRAY, linewidth=1.5, linestyle="--", label="原始單位圓(輸入空間)")
ax2.plot(after_u[0], after_u[1], color=GREEN, linewidth=2.2, label="最終橢圓(A直接作用的結果 = U Σ Vᵀ)")
ax2.axhline(0, color=GRAY, linewidth=0.5)
ax2.axvline(0, color=GRAY, linewidth=0.5)
ax2.set_aspect("equal")
ax2.set_title("最後一步:U(旋轉)把橢圓轉到輸出空間的方向")
ax2.legend(loc="upper left", fontsize=9)
lim2 = max(np.abs(after_u).max() * 1.3, 1.5)
ax2.set_xlim(-lim2, lim2)
ax2.set_ylim(-lim2, lim2)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "svd_final_ellipse.png"), bbox_inches="tight")

print("saved rotate_scale_rotate.png and svd_final_ellipse.png")
