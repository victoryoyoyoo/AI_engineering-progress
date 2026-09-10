"""
Cross-entropy loss曲線: loss = -log(模型對正確答案給的機率)
機率越接近1,loss越接近0;機率越接近0,loss趨近無窮大
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

p = np.linspace(0.01, 1, 300)
loss = -np.log(p)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p, loss, color="#4C72B0", lw=2.5)

highlights = [0.99, 0.5, 0.1, 0.01]
colors = ["#55A868", "#CCB974", "#8172B2", "#C44E52"]
for hp, c in zip(highlights, colors):
    hl = -np.log(hp)
    ax.plot(hp, hl, "o", color=c, markersize=8, zorder=5)
    ax.annotate(f"p={hp}\nloss={hl:.2f}", (hp, hl), textcoords="offset points",
                xytext=(8, 8), fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("模型對正確答案給的機率 p")
ax.set_ylabel("loss = -log(p)")
ax.set_ylim(0, 5)
ax.set_title("Cross-entropy loss:機率越接近1 loss越接近0,越接近0 loss飆高", fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("cross_entropy_loss_curve.png", dpi=150, bbox_inches="tight")
print("saved")
