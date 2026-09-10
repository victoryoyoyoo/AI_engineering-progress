"""
中央極限定理(CLT)示意圖:原始分布是均勻分布(一點都不像鐘形),
但把很多組樣本各自取平均之後,平均值的分布趨近常態分布
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import random
import sys

sys.path.insert(0, "..")
from reference import demonstrate_clt  # noqa: E402

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

random.seed(42)


def roll_die():
    return random.randint(1, 6)


single_rolls = [roll_die() for _ in range(3000)]
averages_of_30 = demonstrate_clt(roll_die, n_samples=30, n_averages=3000)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

ax = axes[0]
ax.hist(single_rolls, bins=6, color="#8172B2", alpha=0.75, edgecolor="black")
ax.set_title("原始分布:單次擲骰(均勻分布,完全不是鐘形)", fontsize=10.5, fontweight="bold")
ax.set_xlabel("點數")

ax = axes[1]
ax.hist(averages_of_30, bins=30, color="#4C72B0", alpha=0.75, edgecolor="black")
ax.set_title("30個一組取平均,重複3000組:趨近常態分布", fontsize=10.5, fontweight="bold")
ax.set_xlabel("平均值")

fig.suptitle("中央極限定理(CLT):不管原始分布長什麼樣,樣本平均值都趨近常態分布", fontsize=12.5, fontweight="bold")
plt.tight_layout()
plt.savefig("clt_demo.png", dpi=150, bbox_inches="tight")
print("saved")
