"""
singular matrix(奇異矩陣)示意圖:det=0,單位正方形被壓扁成一條線,面積歸零、資訊遺失、不可逆
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
A = np.array([[2, 1], [4, 2]])  # det = 4-4 = 0
transformed = unit_square @ A.T

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_square[:, 0], unit_square[:, 1], "--", color="#999999", lw=1.3, label="原始單位正方形(有面積)")
ax.fill(unit_square[:, 0], unit_square[:, 1], color="#999999", alpha=0.15)
ax.plot(transformed[:, 0], transformed[:, 1], color="#C44E52", lw=3, label="A壓扁後(det=0,面積=0)")
ax.plot([0, 4], [0, 2], color="#C44E52", lw=3)

# 一個非零向量被壓成零向量
v = np.array([1, -2])  # A@v = [0,0]
ax.annotate("", xy=(0.02, 0.02), xytext=v, arrowprops=dict(arrowstyle="-|>", color="#8172B2", lw=2, mutation_scale=18))
ax.plot([], [], color="#8172B2", lw=2, label="非零向量 v 被 A 壓成零向量:A@v=[0,0]")
ax.scatter([0], [0], color="black", zorder=5, s=30)

ax.axhline(0, color="#bbbbbb", lw=0.6)
ax.axvline(0, color="#bbbbbb", lw=0.6)
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="upper left")
ax.set_title("singular matrix(奇異矩陣):det=0,空間被壓扁成一條線")

plt.tight_layout()
plt.savefig("singular_matrix.png", dpi=150, bbox_inches="tight")
print("saved")
