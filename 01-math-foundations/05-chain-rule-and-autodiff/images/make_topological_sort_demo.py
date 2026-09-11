"""
拓撲排序示意圖:計算圖a=x1*x2, b=a+1, y=relu(b),排序規則是「小孩一定排在爸媽前面」
下方數字是topo list裡的順序位置
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

nodes = {
    "x1": (1, 3, "#8172B2", 0),
    "x2": (1, 1, "#8172B2", 1),
    "a=x1*x2": (3.7, 2, "#4C72B0", 2),
    "b=a+1": (6.2, 2, "#55A868", 3),
    "y=relu(b)": (8.7, 2, "#C44E52", 4),
}
edges = [("x1", "a=x1*x2"), ("x2", "a=x1*x2"), ("a=x1*x2", "b=a+1"), ("b=a+1", "y=relu(b)")]

for a, b in edges:
    xa, ya, _, _ = nodes[a]
    xb, yb, _, _ = nodes[b]
    ax.annotate("", xy=(xb - 0.75, yb), xytext=(xa + 0.75, ya),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.6, mutation_scale=15))

for name, (x, y, c, order) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.7, facecolor=c, alpha=0.22, edgecolor=c, lw=2.2))
    ax.text(x, y, name, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x, y - 1.05, f"topo順序: {order}", ha="center", fontsize=8.5, color=c)

ax.set_title("拓撲排序(Topological Sort):小孩(左)一定排在爸媽(右)前面")
ax.text(5, 0.15, "backward()把這個順序「倒過來」執行,才會是 y -> b -> a -> x1,x2",
        ha="center", fontsize=9.5)

plt.tight_layout()
plt.savefig("topological_sort.png", dpi=150, bbox_inches="tight")
print("saved")
