"""
共變異數示意圖:正相關/負相關/不相關的散點對照,以及共變異數矩陣的對稱性
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

np.random.seed(1)
n = 150
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

x = np.random.normal(0, 1, n)
y_pos = 0.9 * x + np.random.normal(0, 0.4, n)
cov_pos = np.cov(x, y_pos)[0, 1]
axes[0].scatter(x, y_pos, s=18, color="#4C72B0", alpha=0.6)
axes[0].set_title(f"正相關\ncov={cov_pos:.2f}", fontsize=12, fontweight="bold", color="#4C72B0")

y_neg = -0.9 * x + np.random.normal(0, 0.4, n)
cov_neg = np.cov(x, y_neg)[0, 1]
axes[1].scatter(x, y_neg, s=18, color="#C44E52", alpha=0.6)
axes[1].set_title(f"負相關\ncov={cov_neg:.2f}", fontsize=12, fontweight="bold", color="#C44E52")

y_none = np.random.normal(0, 1, n)
cov_none = np.cov(x, y_none)[0, 1]
axes[2].scatter(x, y_none, s=18, color="#55A868", alpha=0.6)
axes[2].set_title(f"不相關\ncov={cov_none:.2f}", fontsize=12, fontweight="bold", color="#55A868")

for ax in axes:
    ax.set_xlabel("特徵1", fontsize=9)
    ax.set_ylabel("特徵2", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("covariance_demo.png", dpi=150, bbox_inches="tight")
print("saved")
