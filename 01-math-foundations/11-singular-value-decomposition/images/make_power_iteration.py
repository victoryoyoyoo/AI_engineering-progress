import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from plot_style import setup_style, BLUE, RED, GREEN, GRAY

import numpy as np
import matplotlib.pyplot as plt

setup_style()

M = np.array([[3.0, 1.0], [1.0, 2.0]])
eigenvalues, eigenvectors = np.linalg.eigh(M)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
dominant_vec = eigenvectors[:, 0]
minor_vec = eigenvectors[:, 1]

np.random.seed(7)
v = np.array([-0.3, 0.95])
v = v / np.linalg.norm(v)
history = [v.copy()]
for _ in range(6):
    v = M @ v
    v = v / np.linalg.norm(v)
    history.append(v.copy())

fig, ax = plt.subplots(figsize=(6.8, 6.8))

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(theta), np.sin(theta)])
ellipse = M @ circle
ellipse = ellipse / np.abs(ellipse).max() * 1.3
ax.plot(ellipse[0], ellipse[1], color=GRAY, linestyle="--", alpha=0.5,
        label="M對圓形的變形效果(示意)")

for i, vec in enumerate(history):
    alpha = 0.25 + 0.75 * (i / (len(history) - 1))
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=RED, alpha=alpha, lw=1.8))
    ax.text(vec[0] * 1.08, vec[1] * 1.08, f"v{i}", color=RED, alpha=alpha, fontsize=9)

ax.annotate("", xy=dominant_vec * 1.15, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.8))
ax.text(dominant_vec[0] * 1.25, dominant_vec[1] * 1.25,
        f"最大特徵值方向\nλ={eigenvalues[0]:.2f}", color=BLUE, fontsize=10, fontweight="bold")

ax.annotate("", xy=minor_vec * 1.15, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
ax.text(minor_vec[0] * 1.25, minor_vec[1] * 1.25,
        f"次要特徵值方向\nλ={eigenvalues[1]:.2f}", color=GREEN, fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect("equal")
ax.axhline(0, color=GRAY, linewidth=0.6)
ax.axvline(0, color=GRAY, linewidth=0.6)
ax.set_title("Power iteration：v0 → v1 → ... → 收斂到最大特徵值方向")
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "power_iteration.png"), bbox_inches="tight")
print("saved power_iteration.png")
print("history:")
for i, vec in enumerate(history):
    print(f"v{i} = {np.round(vec, 4)}")
