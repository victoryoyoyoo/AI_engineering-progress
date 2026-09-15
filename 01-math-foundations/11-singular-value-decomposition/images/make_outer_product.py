import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt

setup_style()

u = np.array([1, 2])
v = np.array([3, 4])
outer = np.outer(u, v)

fig, axes = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})

ax = axes[0]
ax.imshow(u.reshape(-1, 1), cmap="Blues", vmin=0, vmax=5)
for i, val in enumerate(u):
    ax.text(0, i, str(val), ha="center", va="center", fontsize=14, fontweight="bold")
ax.set_xticks([])
ax.set_yticks(range(len(u)))
ax.set_title("u (2×1直行)", color=BLUE)

ax = axes[1]
ax.imshow(v.reshape(1, -1), cmap="Reds", vmin=0, vmax=5)
for j, val in enumerate(v):
    ax.text(j, 0, str(val), ha="center", va="center", fontsize=14, fontweight="bold")
ax.set_yticks([])
ax.set_xticks(range(len(v)))
ax.set_title("vᵀ (1×2橫列)", color=RED)

ax = axes[2]
im = ax.imshow(outer, cmap="Greens", vmin=0, vmax=outer.max())
for i in range(outer.shape[0]):
    for j in range(outer.shape[1]):
        ax.text(j, i, str(outer[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(outer.shape[1]))
ax.set_yticks(range(outer.shape[0]))
ax.set_title("u vᵀ = 外積結果 (2×2矩陣,rank=1)", color=GREEN)

fig.suptitle("外積(outer product):兩個向量 → 一個秩1矩陣", fontsize=13.5, fontweight="bold")
fig.text(0.5, -0.02,
          "u = [1,2]ᵀ,  v = [3,4]ᵀ  →  (u vᵀ)ᵢⱼ = uᵢ × vⱼ  例如 (2,1)位置 = u₂×v₁ = 2×3 = 6",
          ha="center", fontsize=10, color=GRAY)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "outer_product.png"), bbox_inches="tight")
print("saved outer_product.png")
