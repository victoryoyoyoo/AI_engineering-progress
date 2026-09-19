import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY, GOLD, PURPLE

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

setup_style()

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.4)
ax.axis("off")

ax.text(6.5, 6.1, "Multi-head attention:shape 一路怎麼變(B=2 批, T=8 字, H=4 個頭, D=16, E=H×D=64)",
        ha="center", fontsize=12.5, fontweight="bold")

steps = [
    ("輸入 X", "(B, T, E)", "(2, 8, 64)", "2 句話,每句 8 個字,每字 64 個數字", GRAY, "einsum bte,ek->btk"),
    ("Q、K、V 投影", "(B, T, E)", "(2, 8, 64)", "乘上權重矩陣,shape 不變", BLUE, "reshape + transpose"),
    ("切成 4 個頭", "(B, H, T, D)", "(2, 4, 8, 16)", "64 拆成 4×16,頭的軸搬到前面", GREEN, "einsum bhtd,bhsd->bhts"),
    ("注意力分數", "(B, H, T, T)", "(2, 4, 8, 8)", "每個字對每個字的關注程度", GOLD, "softmax, einsum bhts,bhsd->bhtd"),
    ("加權平均 V", "(B, H, T, D)", "(2, 4, 8, 16)", "用權重把 V 混起來", PURPLE, "transpose + reshape"),
    ("合併 4 個頭", "(B, T, E)", "(2, 8, 64)", "拼回原本的形狀", GREEN, "einsum bte,ek->btk"),
    ("輸出投影", "(B, T, E)", "(2, 8, 64)", "跟輸入 shape 一樣,可以接下一層", RED, ""),
]

for n, (name, sym, num, note, color, op) in enumerate(steps):
    y = 5.35 - n * 0.75
    ax.add_patch(FancyBboxPatch((0.3, y - 0.3), 2.5, 0.6, boxstyle="round,pad=0.04",
                                facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.6))
    ax.text(1.55, y, name, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(3.1, y, sym, va="center", fontsize=12, family="monospace", fontweight="bold")
    ax.text(5.5, y, num, va="center", fontsize=11, family="monospace")
    ax.text(7.5, y, note, va="center", fontsize=10)
    if n < len(steps) - 1 and op:
        ax.text(1.55, y - 0.38, "↓ " + op, ha="center", va="center", fontsize=8, color=GRAY)

plt.savefig(os.path.join(os.path.dirname(__file__), "attention_shapes.png"), bbox_inches="tight")
print("saved")
