#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set some plot parameters

Created on Mon Aug 16 20:41:27 2021

@author: jflynn
"""
from math import sqrt

fig_width_pt = 455.24411  # Got this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5) + 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width / golden_mean  # height in inches
fig_size = [fig_width, fig_height]
plotparams = {
    "backend": "ps",
    "font.size": 10,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 10,
    "figure.figsize": fig_size,
    "savefig.bbox": "tight",
    "legend.numpoints": 1,
    "legend.frameon": True,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.25,
}
