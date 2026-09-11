"""
MLE vs MAP對照圖:丟10次硬幣7次正面,MLE直接算比例,MAP被先驗拉回0.5一點
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

labels = ["先驗信念\nBeta(2,2)均值", "MLE估計\n(只看資料)\n7/10", "MAP估計\n(資料+先驗)\n(2+7)/(2+2+10)"]
values = [0.5, 0.7, 9 / 14]
colors = ["#8172B2", "#C44E52", "#4C72B0"]

fig, ax = plt.subplots(figsize=(7.5, 5.2))
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")

ax.axhline(0.5, color="#999999", lw=1, linestyle="--")
ax.set_ylim(0, 0.85)
ax.set_ylabel("估計的正面機率")
ax.set_title("MLE vs MAP:MAP被先驗往0.5拉回一點,資料量越大先驗影響越小")
ax.grid(alpha=0.3, linestyle="--", axis="y")

plt.tight_layout()
plt.savefig("mle_vs_map.png", dpi=150, bbox_inches="tight")
print("saved")
