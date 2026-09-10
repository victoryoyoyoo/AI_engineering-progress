"""
KL divergence示意圖: H(P)是理論下限,H(P,Q)是實際成本,兩者的差就是KL divergence(浪費掉的部分)
用reference.py真實算出來的數值(true分布 vs good模型)
"""
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "..")
from reference import entropy, cross_entropy, kl_divergence

true_dist = [0.7, 0.2, 0.1]
good_q = [0.65, 0.25, 0.1]

h_true = entropy(true_dist, base=2)
h_cross = cross_entropy(true_dist, good_q, base=2)
kl = kl_divergence(true_dist, good_q, base=2)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.bar(["H(P)\n理論下限", "H(P,Q)\n實際成本"], [h_true, h_cross],
       color=["#4C72B0", "#C44E52"], alpha=0.85, width=0.5)

ax.annotate("", xy=(1.28, h_cross), xytext=(1.28, h_true),
            arrowprops=dict(arrowstyle="<->", color="#55A868", lw=2))
ax.text(1.42, (h_true + h_cross) / 2, f"D_KL(P‖Q)\n= {kl:.4f} bits\n(浪費掉的部分)",
        fontsize=10.5, color="#55A868", fontweight="bold", va="center")

ax.text(0, h_true + 0.05, f"{h_true:.4f}", ha="center", fontsize=10.5, fontweight="bold")
ax.text(1, h_cross + 0.05, f"{h_cross:.4f}", ha="center", fontsize=10.5, fontweight="bold")

ax.set_ylabel("bits")
ax.set_xlim(-0.6, 2.4)
ax.set_ylim(0, max(h_cross, h_true) + 0.3)
ax.set_title("KL Divergence = H(P,Q) - H(P):用Q取代P多浪費的bit數", fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.3, linestyle="--", axis="y")

plt.tight_layout()
plt.savefig("kl_divergence_gap.png", dpi=150, bbox_inches="tight")
print("saved", h_true, h_cross, kl)
