"""
Perplexity示意圖:把交叉熵換算成「像在幾個選項間猶豫」的直覺數字,
對照vocab_size,perplexity<vocab_size才代表比隨機亂猜好
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

vocab_size = 50
random_model_perplexity = 81.23   # 課堂demo實際跑出來的數字(未訓練模型)
good_model_perplexity = 8.0        # 示意:訓練好的模型應該遠低於vocab_size

labels = ["隨機亂猜\n(vocab_size)", "未訓練模型\n(demo實際結果)", "訓練好的模型\n(示意)"]
values = [vocab_size, random_model_perplexity, good_model_perplexity]
colors = ["#8172B2", "#C44E52", "#55A868"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

ax.axhline(vocab_size, color="#8172B2", lw=1.3, linestyle="--", alpha=0.7)
ax.text(2.35, vocab_size + 1.5, "vocab_size=50\n(比隨機亂猜的基準線)", fontsize=9, color="#8172B2")

ax.set_ylabel("Perplexity")
ax.set_ylim(0, 95)
ax.set_title("Perplexity:未訓練模型(81.23)比隨機亂猜(50)還爛,訓練好才會遠低於此", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, linestyle="--", axis="y")

plt.tight_layout()
plt.savefig("perplexity_comparison.png", dpi=150, bbox_inches="tight")
print("saved")
