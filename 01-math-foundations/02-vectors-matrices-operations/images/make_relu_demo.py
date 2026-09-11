"""
relu 示意圖: max(0, x),負數全部砍成0、正數維持原樣,是 relu(W@x+b) 這一層裡的非線性來源
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../../_shared")
from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
setup_style()

x = np.linspace(-4, 4, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(x, y, color="#4C72B0", lw=3)
ax.fill_between(x, y, 0, where=(x < 0), color="#C44E52", alpha=0.25, label="負數輸入 -> 砍成 0")
ax.fill_between(x, y, 0, where=(x >= 0), color="#55A868", alpha=0.25, label="正數輸入 -> 維持原樣")
ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_xlabel("輸入 x (即 W@x+b 算出來的值)", fontsize=10)
ax.set_ylabel("relu(x) = max(0, x)", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(loc="upper left")
ax.set_title("relu:引入非線性,負數砍0、正數不變")

plt.tight_layout()
plt.savefig("relu.png", dpi=150, bbox_inches="tight")
print("saved")
