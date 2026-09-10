"""
GD vs SGD+Momentum vs Adam在Rosenbrock函數上的真實軌跡對照,
直接呼叫reference.py裡的demo_compare_optimizers用到的class,拉真實跑出來的history
"""
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "..")
from reference import GradientDescent, SGDMomentum, Adam, rosenbrock, rosenbrock_gradient, optimize

start = [-1.0, 1.0]
gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start, steps=3000)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start, steps=3000)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start, steps=3000)

X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1, 2, 200))
Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.contour(X, Y, np.log10(Z + 1), levels=25, cmap="Greys", alpha=0.6)

for name, history, color, lw, z in [
    ("GD", gd_history, "#C44E52", 3.2, 3),
    ("SGD+Momentum", sgd_history, "#4C72B0", 2.4, 2),
    ("Adam", adam_history, "#55A868", 1.6, 1),
]:
    arr = np.array(history)
    ax.plot(arr[:, 0], arr[:, 1], color=color, lw=lw, zorder=z,
            label=f"{name} (3000步後loss={rosenbrock(history[-1]):.2e})")
    ax.plot(arr[-1, 0], arr[-1, 1], "o", color=color, markersize=9, zorder=z + 5,
            markeredgecolor="black", markeredgewidth=0.8)

ax.plot(-1, 1, "s", color="black", markersize=9, label="起點")
ax.plot(1, 1, "*", color="gold", markersize=18, markeredgecolor="black", label="最小值 (1,1)")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("GD vs SGD+Momentum vs Adam:同一個Rosenbrock地形,3000步後的真實軌跡", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("optimizer_comparison_rosenbrock.png", dpi=150, bbox_inches="tight")
print("saved")
print("final losses:", rosenbrock(gd_history[-1]), rosenbrock(sgd_history[-1]), rosenbrock(adam_history[-1]))
