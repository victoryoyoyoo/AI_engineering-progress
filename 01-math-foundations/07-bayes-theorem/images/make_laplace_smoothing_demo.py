"""
Laplace smoothing示意圖:一個訓練時沒看過的詞,MLE給它機率0(整個連乘會被污染成0),
加了smoothing之後,機率變成一個很小但非零的數字
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

vocab_size = 20
total_words = 50
words = ["free", "lottery", "meeting", "未出現詞\n(count=0)"]
counts = [8, 5, 3, 0]

no_smooth = [c / total_words if c > 0 else 0 for c in counts]
smoothed = [(c + 1) / (total_words + vocab_size) for c in counts]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(words))
w = 0.35
ax.bar([i - w / 2 for i in x], no_smooth, width=w, color="#C44E52", alpha=0.8, label="沒有smoothing (MLE)")
ax.bar([i + w / 2 for i in x], smoothed, width=w, color="#55A868", alpha=0.8, label="Laplace smoothing (+1)")
ax.set_xticks(list(x))
ax.set_xticklabels(words, fontsize=9.5)
ax.set_ylabel("P(詞|類別)", fontsize=10.5)
ax.grid(alpha=0.3, linestyle="--", axis="y")
ax.legend()
ax.set_title("Laplace smoothing:未出現詞的機率從0變成一個很小的非零值")
ax.annotate("這裡是0!\n連乘會讓整句話的分數直接塌陷成0", (3 - w / 2, 0), textcoords="offset points",
            xytext=(-10, 30), fontsize=8.5, color="#C44E52", ha="center",
            arrowprops=dict(arrowstyle="->", color="#C44E52"))

plt.tight_layout()
plt.savefig("laplace_smoothing.png", dpi=150, bbox_inches="tight")
print("saved")
