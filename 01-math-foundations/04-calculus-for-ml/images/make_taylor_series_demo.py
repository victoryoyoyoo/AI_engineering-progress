"""
泰勒展開示意圖:一階/二階近似跟真正函數的對照,x0附近誤差小、遠離x0誤差變大
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

def f(x):
    return np.sin(x) + 0.15 * x ** 2

def fprime(x0):
    return np.cos(x0) + 0.3 * x0

def fdoubleprime(x0):
    return -np.sin(x0) + 0.3

x0 = 1.0
x = np.linspace(-2, 4, 300)
y = f(x)
y1 = f(x0) + fprime(x0) * (x - x0)
y2 = y1 + 0.5 * fdoubleprime(x0) * (x - x0) ** 2

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(x, y, color="#333333", lw=2.4, label="f(x) 真正函數")
ax.plot(x, y1, "--", color="#4C72B0", lw=2, label="一階近似(梯度下降用這個)")
ax.plot(x, y2, "--", color="#C44E52", lw=2, label="二階近似(牛頓法用這個)")
ax.plot(x0, f(x0), "o", color="#55A868", markersize=9, zorder=5, label=f"展開點 x0={x0}")

ax.set_ylim(-1, 5)
ax.set_xlim(-2, 4)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="upper left")
ax.set_title("泰勒展開:x0附近近似很準,離越遠誤差越大")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("taylor_series_approximation.png", dpi=150, bbox_inches="tight")
print("saved")
