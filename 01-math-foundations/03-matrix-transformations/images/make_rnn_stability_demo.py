"""
RNN穩定性示意圖:eigenvalue反覆相乘(對應RNN反覆套用同一個權重矩陣),
|eigenvalue|>1 指數爆炸、<1 指數消失、=1 保持穩定
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

t = np.arange(0, 21)
cases = {
    "|λ|=1.15 (爆炸 exploding)": (1.15, "#C44E52"),
    "|λ|=1.00 (穩定 stable)": (1.00, "#55A868"),
    "|λ|=0.85 (消失 vanishing)": (0.85, "#4C72B0"),
}

fig, ax = plt.subplots(figsize=(7, 5))
for lab, (lam, c) in cases.items():
    ax.plot(t, lam ** t, color=c, lw=2.6, marker="o", markersize=3.5, label=lab)

ax.set_yscale("log")
ax.set_xlabel("時間步 t(RNN反覆套用權重矩陣的次數)", fontsize=10.5)
ax.set_ylabel("沿eigenvector方向的梯度大小 = λ^t (log scale)", fontsize=10.5)
ax.grid(alpha=0.3, linestyle="--", which="both")
ax.legend()
ax.set_title("RNN梯度穩定性:eigenvalue絕對值決定爆炸/消失/穩定")

plt.tight_layout()
plt.savefig("rnn_stability.png", dpi=150, bbox_inches="tight")
print("saved")
