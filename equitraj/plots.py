"""Static figures for reports (matplotlib, PNG).

Style: thin 2px lines, one y-axis per chart, a legend whenever there are 2+ series,
a light grid. Series colors come in a fixed order (never cycled), so the same series
gets the same color in every figure: ground truth is always slot 1, model runs follow.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#8a8a85"


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def line_plot(path, curves, xlabel, ylabel, title, hline=None, xlim=None, ylim=None):
    """curves: list of (label, x, y). Colors follow list order."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (label, x, y) in enumerate(curves):
        ax.plot(x, y, color=SERIES[k % len(SERIES)], linewidth=2, label=label)
    if hline is not None:
        ax.axhline(hline, color=GRAY, linewidth=1, linestyle="--")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, xlabel, ylabel, title)
    if len(curves) >= 2:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def band_plot(path, x, mean, lo, hi, label, xlabel, ylabel, title, xlim=None):
    """One series with a shaded spread (e.g. min-max over trajectories)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(x, lo, hi, color=SERIES[0], alpha=0.2, linewidth=0, label=f"{label} spread")
    ax.plot(x, mean, color=SERIES[0], linewidth=2, label=label)
    if xlim:
        ax.set_xlim(*xlim)
    _style(ax, xlabel, ylabel, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
