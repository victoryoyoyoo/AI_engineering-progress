"""
gradient checking示意圖:autodiff算出的梯度 vs 數值法(有限差分)算出的梯度,幾乎完全重合,
差異(3.66e-10)遠小於1e-5門檻,證明反向傳播寫對了
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))

ax = axes[0]
vals = [0.15252426, 0.15252426]
bars = ax.bar(["Autodiff\n(反向傳播)", "Numerical\n(有限差分)"], vals, color=["#4C72B0", "#55A868"], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.8f}", ha="center", fontsize=9)
ax.set_ylim(0, 0.19)
ax.set_ylabel("df/dx  (f(x)=tanh(x³+2x+1), x=0.5)", fontsize=9.5)
ax.set_title("兩種方法算出幾乎一樣的梯度", fontsize=11.5, fontweight="bold")

ax = axes[1]
ax.axhline(1e-5, color="#C44E52", lw=2, linestyle="--", label="可接受門檻 1e-5")
ax.scatter([0], [3.66e-10], color="#4C72B0", s=110, zorder=5, label="實際誤差 3.66e-10")
ax.set_yscale("log")
ax.set_ylim(1e-12, 1e-3)
ax.set_xlim(-1, 1)
ax.set_xticks([])
ax.set_ylabel("|autodiff - numerical| (log scale)", fontsize=9.5)
ax.legend()
ax.set_title("誤差遠小於門檻,引擎寫對了", fontsize=11.5, fontweight="bold")

fig.suptitle("Gradient Checking:用數值法驗證autodiff是否正確")
plt.tight_layout()
plt.savefig("gradient_checking.png", dpi=150, bbox_inches="tight")
print("saved")
