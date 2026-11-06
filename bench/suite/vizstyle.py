"""Figure style for the benchmark plots.

A fixed categorical palette in FIXED slot order (never cycled), recessive grid and axes, thin
marks, ink tokens for text (never the series colour), no dual axis. Call `apply_default()` at
the top of a plot script, then draw with CAT[i] / INK[...] / SEQ.

    from vizstyle import apply_default, CAT, INK, SEQ, seq_colors
    apply_default()

`apply_default` is `apply_talk`, the projection-scale style. `apply()` is the paper-scale base
that `apply_talk()` composes.

Palette on a light surface (#fcfcfb), worst adjacent colour-vision-deficiency difference
ΔE 24.2. Slots 2/3/7 (aqua/yellow/magenta) are below 3:1 contrast on the light surface: use
them only where direct labels or a legend carry identity too.
"""
import matplotlib as mpl

# categorical, fixed order (blue, aqua, yellow, green, violet, red, magenta, orange)
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# high-contrast subset (>=3:1 on light) for thin LINE marks, in order
CAT_SAFE = ["#2a78d6", "#e34948", "#eb6834", "#008300", "#4a3aa7", "#0b0b0b"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# sequential blue ramp (light->dark): near-zero .. max
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def seq_colors(n):
    """n samples along the sequential blue ramp, evenly spaced, skipping the too-light end."""
    import numpy as np
    idx = np.linspace(1, len(SEQ) - 1, max(n, 1)).round().astype(int)
    return [SEQ[i] for i in idx]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"], "axes.titlesize": 11, "axes.titleweight": "medium",
        "axes.edgecolor": INK["baseline"], "axes.linewidth": 0.8,
        # The y-axis is never removed. Top/right are dropped as
        # chart junk; left and bottom carry the scale and are pinned True here so no plot
        # script can quietly drop them.
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "grid.color": INK["grid"], "grid.linewidth": 0.6, "axes.grid": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
        "lines.markersize": 6, "figure.titlesize": 13, "figure.titleweight": "medium",
    })


def grid_y(ax):
    """Recessive horizontal grid behind the marks."""
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)


def label_bars(ax, xs, ys, fmt="{:.0f}", dy=0, fs=7.5):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=fs,
                color=INK["secondary"])


def apply_talk():
    """House style at PROJECTION scale, for slides rather than pages.

    Same palette, same ink tokens, same spine rules, same dpi 400 -- everything in `apply()`
    still holds. What changes is only what a listener five metres from a bad projector cannot
    otherwise read: type roughly 1.8x the paper sizes, thicker lines and larger markers.

    TITLE WEIGHT IS NORMAL, not semibold. The reference render this style is measured against
    carries a regular-weight, centred title, and semibold was simply wrong -- large type is
    what makes a projected title readable, and adding weight on top of it only makes the
    heading shout over the marks it is introducing. Fixed here rather than per script, since
    a per-script override is how a style drifts.
    """
    apply()
    mpl.rcParams.update({
        "font.size": 17,
        # Titles are regular weight, centred: semibold is too heavy at this scale and reads as
        # a second, competing mark. Corrected here rather than per script.
        "axes.labelsize": 19, "axes.titlesize": 21, "axes.titleweight": "normal",
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24, "figure.titleweight": "normal",
        "axes.linewidth": 1.4, "grid.linewidth": 0.9,
        "lines.linewidth": 3.0, "lines.markersize": 11,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    })


# The default for every figure: the projection-scale style (see the module docstring).
apply_default = apply_talk
