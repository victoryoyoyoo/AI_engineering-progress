"""
Elbow method示意圖:explained variance ratio隨主成分數量遞減,找斷崖點決定留幾維
"""
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

sys.path.insert(0, "..")
from reference import PCA

np.random.seed(7)
n_samples = 400
n_features = 6
latent = np.random.normal(0, 1, (n_samples, 2))
mixing = np.random.uniform(-1, 1, (2, n_features))
X = latent @ mixing + np.random.normal(0, 0.05, (n_samples, n_features))

pca = PCA(n_components=n_features)
pca.fit(X)
ratios = pca.explained_variance_ratio_
cumulative = np.cumsum(ratios)

fig, ax1 = plt.subplots(figsize=(7.5, 5))
components = np.arange(1, n_features + 1)

ax1.bar(components, ratios, color="#4C72B0", alpha=0.75, label="單個主成分的explained variance ratio")
ax1.set_xlabel("主成分數量", fontsize=11)
ax1.set_ylabel("Explained variance ratio", fontsize=11, color="#4C72B0")
ax1.set_xticks(components)

ax2 = ax1.twinx()
ax2.plot(components, cumulative, "o-", color="#C44E52", linewidth=2, label="累積explained variance")
ax2.set_ylabel("累積比例", fontsize=11, color="#C44E52")
ax2.set_ylim(0, 1.05)

ax1.axvline(x=2, color="#55A868", linestyle="--", alpha=0.7)
ax1.text(2.1, max(ratios) * 0.9, "斷崖點(elbow)\n第2個之後貢獻趨近雜訊", fontsize=9, color="#55A868")

ax1.set_title("Elbow Method:找變異量的斷崖落差,決定留幾維")
fig.tight_layout()
plt.savefig("elbow_method_demo.png", dpi=150, bbox_inches="tight")
print("saved")
for c, r, cum in zip(components, ratios, cumulative):
    print(f"component={c}  ratio={r:.4f}  cumulative={cum:.4f}")
