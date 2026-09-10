"""
One-hot(獨熱編碼)示意圖:正確類別位置是1,其他全部是0,
代回cross-entropy完整公式後,0乘任何數字都是0,只剩下正確類別那一項
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

classes = ["貓", "狗", "鳥", "魚"]
one_hot = [0, 0, 1, 0]  # 真實類別是"鳥"
q = [0.05, 0.10, 0.70, 0.15]  # 模型猜的分布

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

ax = axes[0]
colors = ["#C44E52" if v == 1 else "#cccccc" for v in one_hot]
ax.bar(classes, one_hot, color=colors, edgecolor="black")
ax.set_ylim(0, 1.2)
ax.set_ylabel("P(x) (one-hot,真實分布)", fontsize=10)
ax.set_title("One-hot:真實類別=1,其他全部=0", fontsize=11.5, fontweight="bold")
for i, v in enumerate(one_hot):
    ax.text(i, v + 0.05, str(v), ha="center", fontsize=10, fontweight="bold")

ax = axes[1]
ax.bar(classes, q, color="#4C72B0", edgecolor="black", alpha=0.8)
ax.set_ylim(0, 1.2)
ax.set_ylabel("Q(x) (模型猜的機率)", fontsize=10)
ax.set_title(f"套公式:只有\"鳥\"那項(p=1)有貢獻\nH(P,Q) = -log({q[2]}) = {-np.log(q[2]):.3f} nats", fontsize=10.5, fontweight="bold")
for i, v in enumerate(q):
    ax.text(i, v + 0.05, f"{v}", ha="center", fontsize=9.5)

fig.suptitle("為什麼分類問題的cross-entropy只看正確答案那一項的機率", fontsize=12.5, fontweight="bold")
plt.tight_layout()
plt.savefig("onehot_encoding.png", dpi=150, bbox_inches="tight")
print("saved")
