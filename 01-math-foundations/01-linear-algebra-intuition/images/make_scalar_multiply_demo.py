"""
純量乘法示意圖:同一個向量乘上不同純量,方向不變(負數除外)只改變長度
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

v = np.array([2, 1])
scalars = [1, 2, 0.5, -1]
colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]
labels = ["v (原始)", "2v (長度x2,方向不變)", "0.5v (長度x0.5,方向不變)", "-1v (方向反過來)"]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
for s, c, lab in zip(scalars, colors, labels):
    vec = s * v
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, mutation_scale=22))
    ax.plot([], [], color=c, lw=2.8, label=lab)

ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("純量乘法(scalar multiply):只改變長度,負數會反轉方向", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig("scalar_multiply.png", dpi=150, bbox_inches="tight")
print("saved")
