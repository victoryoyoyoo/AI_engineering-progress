"""
維度詛咒示意圖:隨著維度增加,最近點跟最遠點的距離差距消失
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

np.random.seed(0)
dims = [1, 2, 3, 5, 10, 50, 100, 500, 1000]
ratios = []
n_points = 1000

for d in dims:
    X = np.random.uniform(0, 1, size=(n_points, d))
    origin = np.zeros(d)
    dists = np.linalg.norm(X - origin, axis=1)
    ratio = (dists.max() - dists.min()) / dists.min()
    ratios.append(ratio)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dims, ratios, marker="o", linewidth=2, color="#C44E52")
ax.set_xscale("log")
ax.set_xlabel("維度數(log scale)", fontsize=11)
ax.set_ylabel("(最遠距離 - 最近距離) / 最近距離", fontsize=11)
ax.set_title("維度詛咒:維度越高,近跟遠的差別越消失")
ax.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("curse_of_dimensionality.png", dpi=150, bbox_inches="tight")
print("saved")
for d, r in zip(dims, ratios):
    print(f"dim={d:5d}  ratio={r:.3f}")
