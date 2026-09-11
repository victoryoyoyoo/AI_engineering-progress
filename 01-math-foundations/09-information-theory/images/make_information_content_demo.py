"""
Information content (驚訝程度)長條圖: -log(p),用reference.py真實算出來的數值
"""
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

sys.path.insert(0, "..")
from reference import information_content

events = [
    ("公平銅板正面\np=0.5", 0.5),
    ("骰子擲到6\np=0.167", 1 / 6),
    ("千分之一的事件\np=0.001", 0.001),
    ("必然發生的事\np=1.0", 1.0),
]
labels = [e[0] for e in events]
values = [information_content(e[1], base=2) for e in events]
colors = ["#4C72B0", "#55A868", "#C44E52", "#CCB974"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f} bits", ha="center", fontsize=10.5, fontweight="bold")

ax.set_ylabel("驚訝程度 I(x) = -log2(p)  (bits)")
ax.set_title("Information Content:機率越低,驚訝程度(資訊量)越大")
ax.set_ylim(0, 11)
ax.grid(alpha=0.3, linestyle="--", axis="y")

plt.tight_layout()
plt.savefig("information_content_bars.png", dpi=150, bbox_inches="tight")
print("saved", values)
