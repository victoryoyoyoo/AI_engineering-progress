"""
2D梯度方向示意圖:等高線圖上畫出梯度向量,指向往上爬最快的方向;
梯度下降走它的反方向
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

def f(x, y):
    return (x - 1) ** 2 + 2 * (y - 0.5) ** 2

def grad(x, y):
    return np.array([2 * (x - 1), 4 * (y - 0.5)])

X, Y = np.meshgrid(np.linspace(-2, 4, 200), np.linspace(-2, 3, 200))
Z = f(X, Y)

px, py = -1.0, -1.0
g = grad(px, py)
g_unit = g / np.linalg.norm(g)

fig, ax = plt.subplots(figsize=(6.5, 6))
cs = ax.contour(X, Y, Z, levels=15, cmap="Blues", alpha=0.8)
ax.clabel(cs, inline=True, fontsize=7)

ax.plot(px, py, "o", color="#333333", markersize=8)
ax.annotate("", xy=(px + g_unit[0] * 1.3, py + g_unit[1] * 1.3), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=2.5, mutation_scale=20))
ax.text(px + g_unit[0] * 1.5, py + g_unit[1] * 1.5, "梯度\n(往上爬最快)", color="#C44E52",
        fontsize=10.5, fontweight="bold")

ax.annotate("", xy=(px - g_unit[0] * 1.3, py - g_unit[1] * 1.3), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color="#4C72B0", lw=2.5, mutation_scale=20))
ax.text(px - g_unit[0] * 1.9, py - g_unit[1] * 1.9, "負梯度\n(梯度下降走這邊)", color="#4C72B0",
        fontsize=10.5, fontweight="bold", ha="center")

ax.plot(1, 0.5, "*", color="#55A868", markersize=18, label="最低點 (1, 0.5)")
ax.legend(loc="upper right")
ax.set_title("2D梯度方向:垂直於等高線,指向數值增加最快的方向")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("gradient_direction_2d.png", dpi=150, bbox_inches="tight")
print("saved")
