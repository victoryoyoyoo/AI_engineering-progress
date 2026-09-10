"""
L2正則化=高斯先驗示意圖:高斯先驗機率密度 vs 取負log後變成的懲罰項(w^2形狀),
兩者是同一件事的兩種寫法
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

w = np.linspace(-3, 3, 300)
sigma = 1.0
prior_density = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-w ** 2 / (2 * sigma ** 2))
neg_log_prior = w ** 2 / (2 * sigma ** 2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(w, prior_density, color="#4C72B0", lw=2.4)
axes[0].fill_between(w, prior_density, color="#4C72B0", alpha=0.15)
axes[0].set_title("統計語言:\n以0為中心的高斯先驗 P(w)", fontsize=11.5, fontweight="bold", color="#4C72B0")
axes[0].set_xlabel("w")
axes[0].set_ylabel("密度")
axes[0].grid(alpha=0.3, linestyle="--")

axes[1].plot(w, neg_log_prior, color="#C44E52", lw=2.4)
axes[1].fill_between(w, neg_log_prior, color="#C44E52", alpha=0.15)
axes[1].set_title("工程語言:\n取負log後 = L2懲罰項 w²/(2σ²)", fontsize=11.5, fontweight="bold", color="#C44E52")
axes[1].set_xlabel("w")
axes[1].set_ylabel("-log P(w)  (懲罰項)")
axes[1].grid(alpha=0.3, linestyle="--")

fig.suptitle("L2正則化本質上就是貝氏統計:同一條公式的兩種寫法", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("l2_regularization_as_gaussian_prior.png", dpi=150, bbox_inches="tight")
print("saved")
