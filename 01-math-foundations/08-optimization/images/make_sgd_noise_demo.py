"""
SGD/mini-batch的「雜訊」示意圖:Batch GD走的路徑平滑精確,
SGD/mini-batch因為每步用估計梯度,路徑吵雜,但這個雜訊有機會把optimizer推出淺的局部最小值
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(3)


def f(x):
    return x ** 2


def grad(x):
    return 2 * x


lr = 0.15
x0 = 4.5

# Batch GD:精確梯度,路徑平滑
x_batch = [x0]
for _ in range(25):
    x_batch.append(x_batch[-1] - lr * grad(x_batch[-1]))

# Mini-batch SGD:梯度加上隨機雜訊,路徑吵雜
x_sgd = [x0]
for _ in range(25):
    noisy_grad = grad(x_sgd[-1]) + rng.normal(0, 1.2)
    x_sgd.append(x_sgd[-1] - lr * noisy_grad)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
xs_curve = np.linspace(-5, 5, 200)
ax.plot(xs_curve, f(xs_curve), color="#cccccc", lw=2, zorder=1)

x_batch = np.array(x_batch)
x_sgd = np.array(x_sgd)
ax.plot(x_batch, f(x_batch), "o-", color="#4C72B0", lw=1.8, markersize=4, label="Batch GD(精確梯度,路徑平滑)")
ax.plot(x_sgd, f(x_sgd), "o-", color="#C44E52", lw=1.4, markersize=4, alpha=0.85,
        label="Mini-batch SGD(估計梯度,路徑吵雜但更快脫離平坦區)")

ax.set_xlabel("參數 x", fontsize=10.5)
ax.set_ylabel("loss = f(x)", fontsize=10.5)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=9.5)
ax.set_title("Batch GD vs Mini-batch SGD:雜訊換來跳出平坦區/淺谷的機會", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("sgd_noise.png", dpi=150, bbox_inches="tight")
print("saved")
