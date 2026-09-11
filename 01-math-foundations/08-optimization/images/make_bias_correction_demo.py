"""
Adam bias correction示意圖:m一開始是0,前幾步的原始估計值會被拉低(偏小),
除以(1-beta1^t)做偏差修正後,才是不偏誤的估計
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

beta1 = 0.9
true_gradient = 1.0  # 假設梯度大致穩定在1.0附近
m = 0.0
m_raw, m_corrected, steps = [], [], list(range(1, 31))
for t in steps:
    m = beta1 * m + (1 - beta1) * true_gradient
    m_raw.append(m)
    m_corrected.append(m / (1 - beta1 ** t))

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(steps, m_raw, "o-", color="#C44E52", lw=2, markersize=4, label="m (未修正,前幾步明顯偏小)")
ax.plot(steps, m_corrected, "o-", color="#55A868", lw=2, markersize=4, label="m_hat = m / (1-beta1^t) (偏差修正後)")
ax.axhline(true_gradient, color="#999999", lw=1.4, linestyle="--", label="真實梯度水準 ≈ 1.0")
ax.set_xlabel("訓練步數 t", fontsize=10.5)
ax.set_ylabel("一階矩估計值", fontsize=10.5)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="lower right")
ax.set_title("Adam的Bias Correction:修正m、v在訓練初期的低估")

plt.tight_layout()
plt.savefig("bias_correction.png", dpi=150, bbox_inches="tight")
print("saved")
