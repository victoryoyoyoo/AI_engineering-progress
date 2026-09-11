"""
MLP架構示意圖:Neuron -> Layer -> MLP,對應reference.py裡MLP([2,4,1])的結構
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.axis("off")

layers = [2, 4, 1]
colors = ["#4C72B0", "#55A868", "#C44E52"]
labels = ["輸入層\n(2個數字)", "隱藏層\n(4個Neuron)", "輸出層\n(1個Neuron)"]
xs = [1.5, 4, 6.5]

positions = {}
for li, (n, x, c) in enumerate(zip(layers, xs, colors)):
    ys = [3 + (n - 1) / 2 * 1.1 - i * 1.1 for i in range(n)]
    for i, y in enumerate(ys):
        ax.add_patch(plt.Circle((x, y), 0.35, facecolor=c, alpha=0.25, edgecolor=c, lw=2))
        positions[(li, i)] = (x, y)
    ax.text(x, 5.3, labels[li], ha="center", fontsize=10.5, fontweight="bold", color=c)

for li in range(len(layers) - 1):
    for i in range(layers[li]):
        for j in range(layers[li + 1]):
            x0, y0 = positions[(li, i)]
            x1, y1 = positions[(li + 1, j)]
            ax.plot([x0 + 0.35, x1 - 0.35], [y0, y1], color="#999999", lw=0.7, alpha=0.6, zorder=0)

ax.text(4, 0.6, "每一條連線就是一個weight,每個Neuron自己還有一個bias\n"
                "Neuron: 加權總和+bias再套tanh -> Layer: 一排並排的Neuron -> MLP: 好幾層Layer疊起來",
        ha="center", fontsize=9.5, color="#333333")

ax.set_title("MLP([2, 4, 1]) 架構:Neuron → Layer → MLP")
plt.tight_layout()
plt.savefig("mlp_architecture.png", dpi=150, bbox_inches="tight")
print("saved")
