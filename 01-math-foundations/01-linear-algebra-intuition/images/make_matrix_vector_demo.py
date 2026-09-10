"""
矩陣乘向量示意圖:矩陣的每一列分別跟輸入向量做一次內積,收集成輸出向量
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

W = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]]
x = [1, 0.5, -0.3]
y = [sum(W[i][j] * x[j] for j in range(3)) for i in range(2)]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

colors = ["#4C72B0", "#55A868"]
row_labels = [f"row0 · x = {y[0]:.2f}", f"row1 · x = {y[1]:.2f}"]

ax.text(1.0, 4.4, "W (2x3矩陣)", fontsize=11, fontweight="bold", color="#333333")
for i, (row, c) in enumerate(zip(W, colors)):
    ax.text(0.6, 3.6 - i * 0.9, f"[{row[0]:>5.1f}, {row[1]:>5.1f}, {row[2]:>5.1f}]",
            fontsize=12, family="monospace", color=c, fontweight="bold")

ax.text(4.0, 4.4, "x (輸入向量)", fontsize=11, fontweight="bold", color="#333333")
ax.text(4.0, 3.15, f"[{x[0]}, {x[1]}, {x[2]}]", fontsize=12, family="monospace", color="#8172B2")

for i, (lab, c) in enumerate(zip(row_labels, colors)):
    ax.annotate("", xy=(7.3, 3.6 - i * 0.9), xytext=(5.8, 3.6 - i * 0.9),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, mutation_scale=16))
    ax.text(7.5, 3.6 - i * 0.9, lab, fontsize=11, color=c, va="center", fontweight="bold")

ax.text(0.6, 1.2, f"輸出 y = W @ x = [{y[0]:.2f}, {y[1]:.2f}]  (matrix有幾列 -> 輸出就幾維)",
        fontsize=11.5, color="#333333")

ax.set_title("矩陣乘向量:每一列各自跟輸入向量做內積,收集成輸出", fontsize=13.5, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("matrix_vector_multiply.png", dpi=150, bbox_inches="tight")
print("saved")
