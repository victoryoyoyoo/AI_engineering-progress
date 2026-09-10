"""
broadcasting示意圖:形狀(輸出維度,1)的bias,自動複製延伸去對齊batch維度
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

def grid(x0, y0, rows, cols, color, label, cell=0.55):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(plt.Rectangle((x0 + j * cell, y0 - i * cell), cell, cell,
                                        facecolor=color, alpha=0.25, edgecolor=color, lw=1.3))
    ax.text(x0 + cols * cell / 2, y0 + 0.4, label, ha="center", fontsize=10.5, fontweight="bold", color=color)

grid(0.5, 4.0, 3, 4, "#4C72B0", "weights @ inputs\nshape (3, 4)  <- 3輸出, batch=4")
grid(5.5, 4.0, 3, 1, "#C44E52", "bias\nshape (3, 1)")

ax.annotate("", xy=(5.3, 3.0), xytext=(5.9, 3.0),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6))
ax.text(6.4, 3.3, "broadcasting:\n第2維是1,\n自動複製4次去對齊", fontsize=9.5, color="#555555")

grid(0.5, 1.6, 3, 4, "#55A868", "bias 複製後 (概念上)\nshape (3, 4),每欄都一樣")

ax.set_title("Broadcasting:bias自動延伸去對齊batch維度", fontsize=13.5, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("broadcasting_bias.png", dpi=150, bbox_inches="tight")
print("saved")
