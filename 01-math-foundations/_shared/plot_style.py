import os

import matplotlib
import matplotlib.font_manager as fm

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
PURPLE = "#8172B2"
GOLD = "#CCB974"
GRAY = "#666666"       # reference lines, de-emphasized elements
AXIS_GRAY = "#888888"  # spines / zero-lines

PALETTE = [BLUE, RED, GREEN, PURPLE, GOLD]

TITLE_SIZE = 13        # single-panel figure title / figure suptitle base
SUPTITLE_SIZE = 13.5   # fig.suptitle on multi-panel figures
SUBPLOT_TITLE_SIZE = 11.5  # per-axes title inside a multi-panel figure
LABEL_SIZE = 11
LEGEND_SIZE = 9.5
TICK_SIZE = 9.5

DPI = 150

FIGSIZE_1 = (7, 5)
FIGSIZE_1_SQUARE = (6.5, 6.5)
FIGSIZE_2 = (11, 4.6)
FIGSIZE_3 = (13.5, 4.5)

_CJK_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


def setup_style():
    if os.path.exists(_CJK_FONT_PATH):
        fm.fontManager.addfont(_CJK_FONT_PATH)
    matplotlib.rcParams.update({
        "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
        "axes.unicode_minus": False,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.titlesize": SUPTITLE_SIZE,
        "figure.titleweight": "bold",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXIS_GRAY,
        "axes.linewidth": 0.9,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })
