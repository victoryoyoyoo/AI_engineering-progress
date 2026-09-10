"""
characteristic equation示意圖: det(A-λI) 是關於 λ 的一元二次函式,它的根就是 eigenvalue
"""
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "..")
from reference import eigenvalues_2x2  # noqa: E402

fm.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

A = np.array([[2, 1], [1, 2]])
trace = A[0, 0] + A[1, 1]
det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
lam1, lam2 = sorted(eigenvalues_2x2(A.tolist()))

lam = np.linspace(-1, 5, 400)
f = lam**2 - trace * lam + det

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lam, f, color="#4C72B0", lw=2.6, label="f(λ) = λ² - trace·λ + det")
ax.axhline(0, color="#999999", lw=0.9)
for l, c in zip([lam1, lam2], ["#C44E52", "#55A868"]):
    ax.scatter([l], [0], color=c, s=90, zorder=5)
    ax.annotate(f"λ={l:.1f}\n(eigenvalue)", (l, 0), textcoords="offset points",
                xytext=(0, -32), ha="center", color=c, fontsize=9.5, fontweight="bold")

ax.set_xlabel("λ", fontsize=11)
ax.set_ylabel("det(A - λI)", fontsize=11)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=10)
ax.set_title("characteristic equation:det(A-λI)=0 的根就是 eigenvalue", fontsize=12.5, fontweight="bold")

plt.tight_layout()
plt.savefig("characteristic_equation.png", dpi=150, bbox_inches="tight")
print("saved")
