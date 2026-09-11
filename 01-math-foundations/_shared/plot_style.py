"""
Shared matplotlib style for every make_*.py diagram script under 01-math-foundations.

Import this once near the top of a script and call setup_style() before any
figure is created, so all ten lessons' diagrams share one look instead of
drifting style from session to session.

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
    from plot_style import setup_style, BLUE, RED, GREEN, PURPLE, GOLD, GRAY
    setup_style()
"""
import matplotlib
import matplotlib.font_manager as fm

# ---- color palette --------------------------------------------------------
# One accent color per recurring role. Reused the same way across all
# lessons: BLUE is always "first / primary", RED is always "highlight /
# contrast / warning", GREEN is always "correct / success", PURPLE and GOLD
# are the third and fourth items when a diagram needs more than two.
BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
PURPLE = "#8172B2"
GOLD = "#CCB974"
GRAY = "#666666"       # reference lines, de-emphasized elements
AXIS_GRAY = "#888888"  # spines / zero-lines

PALETTE = [BLUE, RED, GREEN, PURPLE, GOLD]

# ---- typography / sizing ---------------------------------------------------
TITLE_SIZE = 13        # single-panel figure title / figure suptitle base
SUPTITLE_SIZE = 13.5   # fig.suptitle on multi-panel figures
SUBPLOT_TITLE_SIZE = 11.5  # per-axes title inside a multi-panel figure
LABEL_SIZE = 11
LEGEND_SIZE = 9.5
TICK_SIZE = 9.5

DPI = 150

# Common figure sizes, picked by how many panels a diagram needs.
FIGSIZE_1 = (7, 5)
FIGSIZE_1_SQUARE = (6.5, 6.5)
FIGSIZE_2 = (11, 4.6)
FIGSIZE_3 = (13.5, 4.5)

_CJK_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


def setup_style():
    """Apply the shared rcParams. Call this once, before creating any figure."""
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
