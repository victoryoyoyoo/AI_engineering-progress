"""
線性迴歸訓練示意圖:用reference.py同樣的資料跑梯度下降,左圖看擬合線收斂過程,右圖看loss曲線下降
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

import random
random.seed(42)
w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]  # 真實關係 y = 2x + 1

losses = []
snapshots = {}
for epoch in range(200):
    total_loss = 0
    dw = db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs); db /= len(xs); total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    losses.append(total_loss)
    if epoch in (0, 20, 199):
        snapshots[epoch] = (w, b)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.scatter(xs, ys, color="#333333", zorder=5, s=50, label="真實資料 (y=2x+1)")
colors = ["#C44E52", "#CCB974", "#55A868"]
xline = np.array([0, 6])
for (ep, (wv, bv)), c in zip(snapshots.items(), colors):
    ax.plot(xline, wv * xline + bv, color=c, lw=2, label=f"epoch {ep}: y={wv:.2f}x+{bv:.2f}")
ax.set_xlim(0, 6); ax.set_ylim(0, 13)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=8.5)
ax.set_title("擬合線隨訓練逐漸逼近真實關係", fontsize=11.5, fontweight="bold")

ax = axes[1]
ax.plot(losses, color="#4C72B0", lw=2.4)
ax.set_yscale("log")
ax.set_xlabel("epoch", fontsize=10)
ax.set_ylabel("loss (MSE, log scale)", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("loss隨梯度下降穩定下降", fontsize=11.5, fontweight="bold")

fig.suptitle("線性迴歸(Linear Regression):predict -> loss -> gradient -> update", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("linear_regression_training.png", dpi=150, bbox_inches="tight")
print("saved", f"final w={w:.3f} b={b:.3f}")
