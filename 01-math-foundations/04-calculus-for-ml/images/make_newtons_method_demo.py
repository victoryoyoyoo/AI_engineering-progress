"""
牛頓法 vs 梯度下降示意圖:同一個1D函數上,牛頓法(用Hessian)通常幾步就跳到谷底,
梯度下降需要更多步慢慢逼近
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False


def f(x):
    return x ** 2


def f_prime(x):
    return 2 * x


def f_double_prime(x):
    return 2.0


x_curve = np.linspace(-5.5, 5.5, 300)

# 梯度下降:新值 = 舊值 - lr*梯度
gd_path = [5.0]
lr = 0.3
for _ in range(9):
    gd_path.append(gd_path[-1] - lr * f_prime(gd_path[-1]))

# 牛頓法:新值 = 舊值 - 梯度/Hessian
newton_path = [5.0]
for _ in range(9):
    newton_path.append(newton_path[-1] - f_prime(newton_path[-1]) / f_double_prime(newton_path[-1]))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(x_curve, f(x_curve), color="#999999", lw=2, label="f(x) = x²", zorder=1)

gd_path = np.array(gd_path)
newton_path = np.array(newton_path)
ax.plot(gd_path, f(gd_path), "o-", color="#4C72B0", lw=2, markersize=6,
        label=f"梯度下降(9步,只用梯度)")
ax.plot(newton_path, f(newton_path), "o-", color="#C44E52", lw=2, markersize=6,
        label=f"牛頓法(理論上1步到谷底,用梯度+Hessian)")

ax.axhline(0, color="#bbbbbb", lw=0.6)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=9.5, loc="upper center")
ax.set_title("牛頓法 vs 梯度下降:多用二階資訊(Hessian)收斂更快", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("newtons_method_vs_gd.png", dpi=150, bbox_inches="tight")
print("saved")
