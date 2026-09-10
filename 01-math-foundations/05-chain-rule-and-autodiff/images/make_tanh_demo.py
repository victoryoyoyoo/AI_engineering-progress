"""
tanh(雙曲正切)示意圖:把任何數字壓縮成-1到1之間的S形曲線,MLP裡的激活函數
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

x = np.linspace(-4, 4, 400)
y = np.tanh(x)
dy = 1 - y ** 2

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y, color="#4C72B0", lw=2.6, label="tanh(x)")
ax.plot(x, dy, color="#C44E52", lw=2.2, linestyle="--", label="導數: 1 - tanh(x)²")
ax.axhline(1, color="#999999", lw=0.7, linestyle=":")
ax.axhline(-1, color="#999999", lw=0.7, linestyle=":")
ax.axhline(0, color="#bbbbbb", lw=0.7)
ax.axvline(0, color="#bbbbbb", lw=0.7)
ax.set_ylim(-1.3, 1.3)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=10, loc="lower right")
ax.set_title("tanh:壓縮到(-1,1)的S形曲線,中間變化快、兩端變化慢", fontsize=12.5, fontweight="bold")

plt.tight_layout()
plt.savefig("tanh_activation.png", dpi=150, bbox_inches="tight")
print("saved")
