"""
XOR訓練loss曲線:重跑reference.py裡demo_train_xor的邏輯,記錄每一步真實loss畫成曲線
"""
import sys
import random
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "..")
from reference import MLP

random.seed(42)
model = MLP([2, 4, 1])

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]

losses = []
for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    losses.append(loss.data)

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(range(len(losses)), losses, color="#4C72B0", lw=2.3)
ax.scatter([0, 99], [losses[0], losses[-1]], color="#C44E52", s=60, zorder=5)
ax.annotate(f"step 0\nloss={losses[0]:.3f}", (0, losses[0]), textcoords="offset points",
            xytext=(15, -5), fontsize=9.5, color="#C44E52")
ax.annotate(f"step 99\nloss={losses[-1]:.4f}", (99, losses[-1]), textcoords="offset points",
            xytext=(-70, 15), fontsize=9.5, color="#C44E52")

ax.set_ylim(0, 4.6)
ax.set_xlabel("訓練步數 (step)")
ax.set_ylabel("loss (MSE)")
ax.set_title("親手刻的autograd引擎訓練XOR問題:loss真的持續下降", fontsize=12.5, fontweight="bold", pad=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("xor_training_loss.png", dpi=150, bbox_inches="tight")
print("saved, final loss:", losses[-1])
