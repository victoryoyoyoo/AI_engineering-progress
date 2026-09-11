"""
Cross-entropy = Negative log-likelihood示意圖:用reference.py同一組1000筆隨機樣本demo,
兩條完全不同的公式算出一模一樣的數字
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import math
import random
import sys

sys.path.insert(0, "..")
from reference import negative_log_likelihood, softmax  # noqa: E402

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

random.seed(42)
n_samples = 1000
n_classes = 3
labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
all_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_avg = negative_log_likelihood(labels, all_logits)
nll_avg = -sum(
    math.log(softmax(lg)[lb])
    for lb, lg in zip(labels, all_logits)
) / n_samples

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(["Cross-entropy loss\n(資訊理論角度)", "Negative log-likelihood\n(統計/MLE角度)"],
              [ce_avg, nll_avg], color=["#4C72B0", "#55A868"], width=0.5)
for b, v in zip(bars, [ce_avg, nll_avg]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.6f}", ha="center", fontsize=10)
ax.set_ylabel("平均loss (nats)", fontsize=10.5)
ax.set_ylim(0, max(ce_avg, nll_avg) * 1.3)
ax.set_title(f"兩條不同來源的公式,算出完全相同的數字\n(1000筆樣本,差異={abs(ce_avg-nll_avg):.2e})")

plt.tight_layout()
plt.savefig("ce_nll_equality.png", dpi=150, bbox_inches="tight")
print("saved", ce_avg, nll_avg)
