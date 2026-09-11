"""
Eigendecomposition流水線示意圖: A = V @ D @ V^-1
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, ax = plt.subplots(figsize=(13, 4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

steps = [
    ("V⁻¹", "換到eigenvector座標系\n(把座標軸轉成eigenvector方向)", "#4C72B0"),
    ("D", "沿各軸用對應eigenvalue伸縮\n(對角矩陣,各自獨立縮放)", "#55A868"),
    ("V", "換回原本座標系\n(轉回來)", "#C44E52"),
]

box_w, box_h = 3.3, 2.6
gap = 0.5
x0 = 0.4

for i, (title, desc, color) in enumerate(steps):
    x = x0 + i * (box_w + gap)
    y = 0.6
    box = FancyBboxPatch(
        (x, y), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.15
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y + box_h - 0.45, title, ha="center", va="top",
            fontsize=16, fontweight="bold", color=color)
    ax.text(x + box_w / 2, y + box_h - 1.15, desc, ha="center", va="top",
            fontsize=9.5, color="#333333", linespacing=1.6)

    if i < len(steps) - 1:
        arrow = FancyArrowPatch(
            (x + box_w + 0.03, y + box_h / 2),
            (x + box_w + gap - 0.03, y + box_h / 2),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#555555"
        )
        ax.add_patch(arrow)

ax.text(12.5, 2.1, "= A", fontsize=18, fontweight="bold", color="#333333", va="center")

ax.set_title("Eigendecomposition: A = V @ D @ V⁻¹  (輸入向量從右邊套用起)", pad=14)
plt.tight_layout()
plt.savefig("eigendecomposition_pipeline.png", dpi=150, bbox_inches="tight")
print("saved")
