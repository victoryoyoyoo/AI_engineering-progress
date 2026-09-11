"""
貝氏定理示意圖:Prior(先驗)乘上Likelihood(似然)、除以Evidence(證據),得到Posterior(後驗)
用flow示意四個量各自扮演的角色
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, ax = plt.subplots(figsize=(9.5, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

boxes = [
    (0.3, "Prior\nP(A)\n看到證據前的初始信念", "#8172B2"),
    (2.6, "Likelihood\nP(B|A)\n假設A成立,看到B的機率", "#4C72B0"),
]
for x, text, c in boxes:
    ax.add_patch(plt.Rectangle((x, 1.3), 2.1, 1.6, facecolor=c, alpha=0.22, edgecolor=c, lw=2))
    ax.text(x + 1.05, 2.1, text, ha="center", va="center", fontsize=8.8, fontweight="bold", color=c)

ax.text(4.95, 2.1, "×", fontsize=22, ha="center", va="center", fontweight="bold")
ax.text(6.35, 2.1, "÷", fontsize=22, ha="center", va="center", fontweight="bold")

ax.add_patch(plt.Rectangle((6.9, 1.3), 2.1, 1.6, facecolor="#CCB974", alpha=0.3, edgecolor="#CCB974", lw=2))
ax.text(7.95, 2.1, "Evidence\nP(B)\nB本身發生的總機率\n(正規化用的分母)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#8B7B2E")

ax.annotate("", xy=(5, 0.2), xytext=(5, 1.2), arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=2.4, mutation_scale=18))
ax.add_patch(plt.Rectangle((3.3, -0.9), 3.4, 1.05, facecolor="#C44E52", alpha=0.25, edgecolor="#C44E52", lw=2))
ax.text(5, -0.4, "Posterior  P(A|B)\n看到證據後,更新過的信念", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#C44E52")

ax.set_ylim(-1.3, 3.2)
ax.set_title("貝氏定理:Prior × Likelihood ÷ Evidence = Posterior")

plt.tight_layout()
plt.savefig("bayes_theorem_flow.png", dpi=150, bbox_inches="tight")
print("saved")
