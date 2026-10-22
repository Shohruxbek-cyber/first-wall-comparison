"""Depth grid for the He model: 1 nm cells at the surface, geometric growth (x1.065), uniform 1 um
cells out to 50 um, then geometric coarsening to the back of the 1 mm armour.  Surface
recession by erosion (~1e-4 nm/shot) is ignored."""
from __future__ import annotations

import numpy as np


def make_grid(dx0=1e-9, growth=1.065, dx_mid=1.0e-6, x_fine=50e-6, growth_back=1.12, L=1e-3):
    """Return cell edges [m] (len n+1)."""
    dx, x = [], 0.0
    d = dx0
    while d < dx_mid:                       # surface layer: geometric growth
        dx.append(d); x += d; d *= growth
    while x < x_fine:                       # resolved region
        dx.append(dx_mid); x += dx_mid
    d = dx_mid
    while x < L - 0.5 * d:                  # coarse region
        d *= growth_back
        d = min(d, L - x)
        dx.append(d); x += d
    dx[-1] += L - x                         # close exactly at L
    return np.r_[0.0, np.cumsum(dx)]


def refine(edges, factor=2):
    """Grid with every cell split into `factor` equal cells (convergence test)."""
    out = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        out.extend(np.linspace(a, b, factor + 1)[1:])
    return np.array(out)
