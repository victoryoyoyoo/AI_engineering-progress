"""
Forward mode vs reverse mode示意圖:種子放在哪裡、往哪個方向傳
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

nodes = ["x", "a", "b", "y"]

def draw_chain(ax, title, color, arrow_forward, seed_text, seed_at, label):
    xs = [1, 3, 5, 7]
    for x, n in zip(xs, nodes):
        box = FancyBboxPatch((x - 0.5, 0.3), 1, 1, boxstyle="round,pad=0.05,rounding_size=0.1",
                              linewidth=1.6, edgecolor="#555555", facecolor="#EEEEEE")
        ax.add_patch(box)
        ax.text(x, 0.8, n, ha="center", va="center", fontsize=13, fontweight="bold")
    for i in range(3):
        if arrow_forward:
            arrow = FancyArrowPatch((xs[i] + 0.5, 0.8), (xs[i + 1] - 0.5, 0.8),
                                     arrowstyle="-|>", mutation_scale=16, color=color, lw=2)
        else:
            arrow = FancyArrowPatch((xs[i + 1] - 0.5, 0.8), (xs[i] + 0.5, 0.8),
                                     arrowstyle="-|>", mutation_scale=16, color=color, lw=2)
        ax.add_patch(arrow)
    ax.text(xs[seed_at], 1.7, seed_text, ha="center", fontsize=10.5, color=color, fontweight="bold")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2.3)
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=color)
    ax.text(4, -0.2, label, ha="center", fontsize=9.5, color="#555555")

draw_chain(axes[0], "Forward Mode\n種子放輸入,往前推導數", "#4C72B0",
           True, "種子: dx/dx=1", 0, "適合:輸入少、輸出多")
draw_chain(axes[1], "Reverse Mode\n種子放輸出,往回拉梯度", "#C44E52",
           False, "種子: dy/dy=1", 3, "適合:輸入多、輸出少 (神經網路用這個)")

fig.suptitle("Forward Mode vs Reverse Mode:種子放的位置跟傳播方向不同", fontsize=13.5, fontweight="bold")
plt.tight_layout()
plt.savefig("forward_vs_reverse_mode.png", dpi=150, bbox_inches="tight")
print("saved")
