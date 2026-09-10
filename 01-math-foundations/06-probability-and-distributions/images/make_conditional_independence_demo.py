"""
sample space/event/conditional probability/independence 示意圖:
左圖用文氏圖畫出樣本空間、事件A、事件B,P(A|B)=A∩B的面積佔B的比例
右圖對照獨立事件(A的圓完全不受B是否發生影響)
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))

# 左圖:條件機率,A、B有重疊
ax = axes[0]
ax.add_patch(patches.Rectangle((0, 0), 10, 8, facecolor="#f0f0f0", edgecolor="black", lw=1.5))
ax.text(0.3, 7.5, "樣本空間 S(所有可能結果)", fontsize=9)
circA = patches.Circle((4, 4), 2.6, facecolor="#4C72B0", alpha=0.4, edgecolor="#4C72B0", lw=2)
circB = patches.Circle((6, 4), 2.6, facecolor="#C44E52", alpha=0.4, edgecolor="#C44E52", lw=2)
ax.add_patch(circA); ax.add_patch(circB)
ax.text(2.6, 4, "事件A", fontsize=11, fontweight="bold", color="#4C72B0", ha="center")
ax.text(7.4, 4, "事件B", fontsize=11, fontweight="bold", color="#C44E52", ha="center")
ax.text(5, 4, "A∩B", fontsize=9.5, ha="center", fontweight="bold")
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("條件機率 P(A|B) = P(A∩B) / P(B)\n(只看B這個圈裡,A佔了多少比例)", fontsize=10.5, fontweight="bold")

# 右圖:獨立事件,重疊比例跟A本身的大小比例一致(用不重疊示意「互不影響」的簡化畫法)
ax = axes[1]
ax.add_patch(patches.Rectangle((0, 0), 10, 8, facecolor="#f0f0f0", edgecolor="black", lw=1.5))
ax.text(0.3, 7.5, "樣本空間 S", fontsize=9)
circA2 = patches.Circle((3.2, 4), 2.2, facecolor="#55A868", alpha=0.4, edgecolor="#55A868", lw=2)
circB2 = patches.Circle((6.8, 4), 2.2, facecolor="#CCB974", alpha=0.4, edgecolor="#CCB974", lw=2)
ax.add_patch(circA2); ax.add_patch(circB2)
ax.text(3.2, 4, "事件A", fontsize=11, fontweight="bold", color="#55A868", ha="center")
ax.text(6.8, 4, "事件B", fontsize=11, fontweight="bold", color="#8172B2", ha="center")
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("獨立事件(Independence)\nP(A|B) = P(A):知道B發生不改變A的機率", fontsize=10.5, fontweight="bold")

plt.tight_layout()
plt.savefig("conditional_independence.png", dpi=150, bbox_inches="tight")
print("saved")
