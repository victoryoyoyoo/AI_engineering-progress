import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, GOLD

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

setup_style()

fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.6)
ax.axis("off")

ax.text(6.5, 5.3, "einsum 規則:箭頭右邊沒出現的字母 → 那個軸被「乘完加總」掉", ha="center", fontsize=13, fontweight="bold")

rows = [
    ("i,i->", "內積(dot)", "i 出現在左邊,右邊沒有 → 沿 i 相乘再加總",
     "[1,2,3]·[4,5,6] = 1×4+2×5+3×6 = 32", BLUE),
    ("ik,kj->ij", "矩陣乘法", "k 右邊沒有 → 沿 k 相乘再加總;i、j 留下",
     "[[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]", GREEN),
    ("ij->ji", "轉置", "沒有字母消失,只是 i、j 換位置",
     "shape (2,3) → (3,2),數字都留著", RED),
    ("ij->i", "每列加總", "j 右邊沒有 → 沿 j 加總;i 留下",
     "[[1,2,3],[4,5,6]] → [6, 15]", GOLD),
]
for n, (expr, name, why, ex, color) in enumerate(rows):
    y = 4.3 - n * 1.15
    ax.add_patch(FancyBboxPatch((0.3, y - 0.4), 2.7, 0.8, boxstyle="round,pad=0.05",
                                facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.8))
    ax.text(1.65, y, expr, ha="center", va="center", fontsize=15, fontweight="bold", family="monospace")
    ax.text(3.3, y + 0.18, name, fontsize=12, fontweight="bold", va="center")
    ax.text(3.3, y - 0.2, why, fontsize=10, va="center")
    ax.text(9.0, y, ex, fontsize=10, va="center")

plt.savefig(os.path.join(os.path.dirname(__file__), "einsum_rules.png"), bbox_inches="tight")
print("saved")
