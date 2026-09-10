"""
產生PCA fit方法的5步驟流水線示意圖
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis("off")

steps = [
    ("1. 置中\nCenter", "X - mean(X)\n把資料中心移到原點", "#4C72B0"),
    ("2. 共變異數矩陣\nCovariance", "np.cov(X, rowvar=False)\n每對特徵一起變動的程度", "#55A868"),
    ("3. 特徵分解\nEigendecomp", "np.linalg.eigh(cov)\n拆成方向+變異量大小", "#C44E52"),
    ("4. 排序\nSort", "argsort(eigenvalues)[::-1]\n變異量由大到小排", "#8172B2"),
    ("5. 留前k個\nKeep top-k", "components = eigenvectors[:,:k]\n只留變異量最大的k個方向", "#CCB974"),
]

box_w, box_h = 2.15, 2.6
gap = 0.35
x0 = 0.15

for i, (title, desc, color) in enumerate(steps):
    x = x0 + i * (box_w + gap)
    y = 0.6
    box = FancyBboxPatch(
        (x, y), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.15
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y + box_h - 0.45, title, ha="center", va="top",
             fontsize=11.5, fontweight="bold", color=color)
    ax.text(x + box_w / 2, y + box_h - 1.15, desc, ha="center", va="top",
             fontsize=8.3, color="#333333", linespacing=1.6, wrap=True)

    if i < len(steps) - 1:
        arrow = FancyArrowPatch(
            (x + box_w + 0.03, y + box_h / 2),
            (x + box_w + gap - 0.03, y + box_h / 2),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#555555"
        )
        ax.add_patch(arrow)

ax.set_title("PCA.fit() 五步驟流水線", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("pca_fit_pipeline.png", dpi=150, bbox_inches="tight")
print("saved")
