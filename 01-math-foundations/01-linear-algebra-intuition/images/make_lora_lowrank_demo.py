"""
LoRA(Low-Rank Adaptation) 示意圖:把一個大的更新矩陣 ΔW(m x n) 拆成兩個小矩陣 B(m x r)、A(r x n) 相乘
只訓練 B、A,參數量從 m*n 大幅降到 (m+n)*r
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

m, n, r = 6, 6, 1.2  # r 畫得很窄,強調「低秩」

fig, ax = plt.subplots(figsize=(9, 4.5))

def draw_box(x, y, w, h, color, label, alpha=0.75):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                                    lw=1.3, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold")

# ΔW: 大矩陣,參數量 m*n = 36
draw_box(0, 0, m, n, "#C44E52", "ΔW\n(6x6 = 36個參數)")
ax.text(m / 2, n + 0.4, "微調需要的權重更新量,形狀跟原矩陣一樣大\n但實際上「有用的方向」很少(rank 很低)",
        ha="center", fontsize=9.5)

ax.text(m + 1.0, n / 2, "≈", fontsize=26, ha="center", va="center", fontweight="bold")

# B: m x r
bx = m + 2.2
draw_box(bx, 0, r, n, "#4C72B0", "B\n(6x1)")
ax.text(bx + 1.6, n / 2, "@", fontsize=22, ha="center", va="center", fontweight="bold")

# A: r x n
ax_x = bx + 2.6
draw_box(ax_x, n - r, n, r, "#55A868", "A (1x6)")

ax.text(ax_x, -1.1,
        "只訓練 B、A 兩個小矩陣:參數量從 m*n=36 降到 (m+n)*r=14\nLoRA 靠「低秩」假設,大幅省下微調要訓練的參數量",
        fontsize=9.5, ha="left")

ax.set_xlim(-0.5, ax_x + n + 0.5)
ax.set_ylim(-2.2, n + 1.2)
ax.axis("off")
ax.set_title("LoRA(Low-Rank Adaptation):ΔW = B @ A,低秩分解", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("lora_lowrank.png", dpi=150, bbox_inches="tight")
print("saved")
