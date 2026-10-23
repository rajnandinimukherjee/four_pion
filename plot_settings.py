"""Shared Matplotlib settings for the analysis figures."""

from __future__ import annotations

from math import sqrt
from typing import Any


FIG_WIDTH_PT: float = 455.24411  # LaTeX column width, from ``\showthe\columnwidth``.
INCHES_PER_PT: float = 1.0 / 72.27
GOLDEN_MEAN: float = (sqrt(5.0) + 1.0) / 2.0
FIG_WIDTH: float = FIG_WIDTH_PT * INCHES_PER_PT
FIG_HEIGHT: float = FIG_WIDTH / GOLDEN_MEAN
FIG_SIZE: list[float] = [FIG_WIDTH, FIG_HEIGHT]

# Original names remain available to notebooks that imported them directly.
fig_width_pt: float = FIG_WIDTH_PT
inches_per_pt: float = INCHES_PER_PT
golden_mean: float = GOLDEN_MEAN
fig_width: float = FIG_WIDTH
fig_height: float = FIG_HEIGHT
fig_size: list[float] = FIG_SIZE

# Keep the historical lowercase name used throughout the analysis scripts.
plotparams: dict[str, Any] = {
    "backend": "ps",
    "font.size": 10,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 10,
    "figure.figsize": FIG_SIZE,
    "savefig.bbox": "tight",
    "legend.numpoints": 1,
    "legend.frameon": True,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.25,
}
