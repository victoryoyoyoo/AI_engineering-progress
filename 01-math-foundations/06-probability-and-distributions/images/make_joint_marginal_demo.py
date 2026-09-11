"""
Joint / Marginal分布示意圖:聯合機率表格,每一列/每一欄加總變成marginal
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

joint = np.array([[0.10, 0.05, 0.05],
                   [0.15, 0.20, 0.05],
                   [0.05, 0.10, 0.25]])
row_labels = ["X=下雨", "X=多雲", "X=晴天"]
col_labels = ["Y=帶傘", "Y=看情況", "Y=不帶傘"]
marginal_x = joint.sum(axis=1)
marginal_y = joint.sum(axis=0)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
n, m = joint.shape

for i in range(n):
    for j in range(m):
        ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor="#4C72B0",
                                    alpha=0.15 + joint[i, j] * 2, edgecolor="#4C72B0", lw=1))
        ax.text(j + 0.5, n - 1 - i + 0.5, f"{joint[i, j]:.2f}", ha="center", va="center", fontsize=11)

for j in range(m):
    ax.add_patch(plt.Rectangle((j, n), 1, 0.8, facecolor="#55A868", alpha=0.25, edgecolor="#55A868", lw=1))
    ax.text(j + 0.5, n + 0.4, f"{marginal_y[j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color="#55A868")

for i in range(n):
    ax.add_patch(plt.Rectangle((m, n - 1 - i), 0.8, 1, facecolor="#C44E52", alpha=0.25, edgecolor="#C44E52", lw=1))
    ax.text(m + 0.4, n - 1 - i + 0.5, f"{marginal_x[i]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color="#C44E52")

for j, lab in enumerate(col_labels):
    ax.text(j + 0.5, -0.3, lab, ha="center", fontsize=9.5)
for i, lab in enumerate(row_labels):
    ax.text(-0.15, n - 1 - i + 0.5, lab, ha="right", va="center", fontsize=9.5)

ax.text(m + 0.4, n + 0.4, "1.00", ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")

ax.text(m / 2, n + 1.1, "Marginal P(Y): 每一欄加總", ha="center", fontsize=10, color="#55A868", fontweight="bold")
ax.text(m + 1.6, n / 2, "Marginal\nP(X):\n每一列\n加總", ha="center", fontsize=10, color="#C44E52", fontweight="bold")

ax.set_xlim(-1.3, m + 2.3)
ax.set_ylim(-0.8, n + 1.6)
ax.axis("off")
ax.set_title("Joint distribution P(X,Y) 跟 Marginal distribution 的關係")

plt.tight_layout()
plt.savefig("joint_marginal_distribution.png", dpi=150, bbox_inches="tight")
print("saved")
