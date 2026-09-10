"""
鞍點示意圖:3D曲面對照,鞍點某個方向往下彎、另一個方向往上彎,
梯度=0但不是真正最低點
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

X, Y = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
Z_min = X ** 2 + Y ** 2          # Hessian eigenvalues全正 -> 真正最低點
Z_saddle = X ** 2 - Y ** 2        # Hessian eigenvalues一正一負 -> 鞍點

fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, Z_min, cmap="Blues", alpha=0.85, edgecolor="none")
ax1.scatter([0], [0], [0], color="#C44E52", s=60, depthshade=False)
ax1.set_title("真正最低點\nHessian eigenvalue全正", fontsize=11.5, fontweight="bold", color="#4C72B0")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z_saddle, cmap="Reds", alpha=0.85, edgecolor="none")
ax2.scatter([0], [0], [0], color="#4C72B0", s=60, depthshade=False)
ax2.set_title("鞍點 (Saddle Point)\nHessian eigenvalue有正有負", fontsize=11.5, fontweight="bold", color="#C44E52")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f")

fig.suptitle("梯度=0的兩種情況:真正最低點 vs 鞍點,靠Hessian的eigenvalue分辨", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("saddle_point_vs_minimum.png", dpi=150, bbox_inches="tight")
print("saved")
