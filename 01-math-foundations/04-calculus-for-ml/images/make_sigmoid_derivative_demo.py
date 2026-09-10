"""
sigmoid與其導數示意圖: f'(x) = f(x)(1-f(x)),導數在f(x)=0.5(x=0)時最大,兩端趨近0
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.linspace(-8, 8, 400)
f = 1 / (1 + np.exp(-x))
f_prime = f * (1 - f)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, f, color="#4C72B0", lw=2.6, label="sigmoid: f(x) = 1/(1+e^-x)")
ax.plot(x, f_prime, color="#C44E52", lw=2.6, label="導數: f'(x) = f(x)(1-f(x))")
ax.axhline(0, color="#bbbbbb", lw=0.7)
ax.axvline(0, color="#bbbbbb", lw=0.7, linestyle="--")
ax.scatter([0], [0.25], color="#C44E52", zorder=5, s=50)
ax.annotate("x=0時導數最大=0.25", (0, 0.25), textcoords="offset points", xytext=(15, 10), fontsize=9.5)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="upper left")
ax.set_title("sigmoid 與導數:f'(x)可以完全用f(x)自己表示", fontsize=12.5, fontweight="bold")

plt.tight_layout()
plt.savefig("sigmoid_derivative.png", dpi=150, bbox_inches="tight")
print("saved")
