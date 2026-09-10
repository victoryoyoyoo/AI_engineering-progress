"""
Expected value(期望值)與 Variance(變異數) 示意圖:骰子範例,用reference.py同一組數字
E[X]=3.5是機率加權平均(不一定要是骰子能擲出的值),Var(X)是每個結果離E[X]有多分散
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "..")
from reference import expected_value, variance  # noqa: E402

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1 / 6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(die_values, die_probs, color="#4C72B0", alpha=0.75, edgecolor="black", width=0.6, label="P(X=x) = 1/6")
ax.axvline(mu, color="#C44E52", lw=2.4, linestyle="--", label=f"E[X] = {mu:.1f} (機率加權平均)")
ax.annotate("", xy=(mu + var ** 0.5, 0.05), xytext=(mu - var ** 0.5, 0.05),
            arrowprops=dict(arrowstyle="<->", color="#55A868", lw=2))
ax.text(mu, 0.02, f"±1個標準差(SD={var**0.5:.2f}) ,Var(X)={var:.2f}", ha="center", fontsize=9, color="#55A868")

ax.set_xlabel("骰子點數 x", fontsize=10.5)
ax.set_ylabel("機率", fontsize=10.5)
ax.set_ylim(0, 0.22)
ax.grid(alpha=0.3, linestyle="--", axis="y")
ax.legend(fontsize=10, loc="upper right")
ax.set_title("Expected value(期望值)與Variance(變異數):骰子範例", fontsize=12.5, fontweight="bold")

plt.tight_layout()
plt.savefig("expected_variance.png", dpi=150, bbox_inches="tight")
print("saved", mu, var)
