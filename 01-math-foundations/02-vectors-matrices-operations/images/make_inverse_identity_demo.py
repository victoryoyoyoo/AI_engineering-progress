"""
inverse(逆矩陣) 與 identity matrix(單位矩陣) 示意圖:
A 把 X 搬到 Y,A_inv 把 Y 搬回 X,A @ A_inv = identity(不改變任何東西的矩陣)
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

A = np.array([[2.0, 0.5], [0.3, 1.5]])
A_inv = np.linalg.inv(A)
x = np.array([1.0, 1.0])
y = A @ x
x_back = A_inv @ y

fig, ax = plt.subplots(figsize=(6.5, 6.5))
for v, c, lab in zip([x, y, x_back], ["#4C72B0", "#C44E52", "#55A868"],
                      ["x (原始)", "y = A @ x", "A⁻¹ @ y = x (搬回原點)"]):
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.6, mutation_scale=20,
                                 linestyle="--" if lab.startswith("A⁻¹") else "-"))
    ax.plot([], [], color=c, lw=2.6, label=lab)

ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlim(-0.5, 3); ax.set_ylim(-0.5, 3)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_title("inverse(逆矩陣):A⁻¹ 把 A 搬過去的東西「搬回來」\nA @ A⁻¹ = identity matrix(單位矩陣,乘上任何東西不改變它)")

check = A @ A_inv
ax.text(0.05, -0.35, f"驗證: A @ A⁻¹ ≈\n{np.round(check, 2)}", transform=ax.transAxes,
        fontsize=9.5)

plt.tight_layout()
plt.savefig("inverse_identity.png", dpi=150, bbox_inches="tight")
print("saved")
