"""Temporal pulse shapes and generic deposition sources for the thermal solver.

A source delivers an energy fluence [J/m^2] to the surface with a time shape, and
distributes it over cells according to a depth profile computed on the *current* mesh
(so profiles follow the receding surface and cross layer boundaries correctly).

Depth is measured in areal mass xi [kg/m^2] from the current surface, which is the
natural variable for photon attenuation (mu/rho) and ion stopping (S/rho).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .thermal import SourceTerm


# --------------------------------------------------------------------------------------
# Temporal shapes (normalised: integral over all time = 1)
# --------------------------------------------------------------------------------------
@dataclass
class SquarePulse:
    t_start: float  # s
    duration: float  # s

    def cumulative(self, t):
        return float(np.clip((t - self.t_start) / self.duration, 0.0, 1.0))

    def active(self, t):
        return self.t_start <= t < self.t_start + self.duration

    def events(self):
        return [self.t_start, self.t_start + self.duration]


@dataclass
class TabulatedPulse:
    """Piecewise-linear power shape on a time grid (e.g. ion time-of-flight arrival)."""

    t: np.ndarray  # s, increasing
    p: np.ndarray  # arbitrary units, >= 0

    def __post_init__(self):
        self.t, self.p = np.asarray(self.t, float), np.asarray(self.p, float)
        seg = 0.5 * (self.p[1:] + self.p[:-1]) * np.diff(self.t)
        self._cum = np.r_[0.0, np.cumsum(seg)]
        self._total = self._cum[-1]

    def cumulative(self, t):
        if t <= self.t[0]:
            return 0.0
        if t >= self.t[-1]:
            return 1.0
        i = np.searchsorted(self.t, t) - 1
        tau = t - self.t[i]
        dt = self.t[i + 1] - self.t[i]
        slope = (self.p[i + 1] - self.p[i]) / dt
        return float((self._cum[i] + self.p[i] * tau + 0.5 * slope * tau ** 2) / self._total)

    def active(self, t):
        return self.t[0] <= t < self.t[-1]

    def events(self):
        return [self.t[0], self.t[-1]]


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------
def cell_mass_bounds(solver):
    """Cumulative areal mass at cell boundaries measured from the current surface, kg/m^2."""
    return np.r_[0.0, np.cumsum(solver.m)]


class PulsedSource(SourceTerm):
    """Energy fluence F [J/m^2] with a temporal shape; subclasses provide `fractions`."""

    def __init__(self, fluence, shape, dt_fraction=0.1):
        self.fluence, self.shape, self.dt_fraction = fluence, shape, dt_fraction

    def energy(self, t0, t1):
        return self.fluence * (self.shape.cumulative(t1) - self.shape.cumulative(t0))

    def dt_limit(self, t):
        if isinstance(self.shape, SquarePulse) and self.shape.active(t):
            return self.dt_fraction * self.shape.duration
        return np.inf

    def next_event(self, t):
        ev = [e for e in self.shape.events() if e > t * (1 + 1e-14) + 1e-300]
        return min(ev) if ev else np.inf


class BeerLawSource(PulsedSource):
    """Monoenergetic photons: absorbed fraction in cell i = exp(-tau_i) - exp(-tau_{i+1}),
    tau = sum over crossed cells of (mu/rho)_material * m_cell.

    mu_rho_by_layer: attenuation coefficient [m^2/kg] per layer index (0 = no absorption)."""

    def __init__(self, fluence, shape, mu_rho_by_layer: Sequence[float], dt_fraction=0.1):
        super().__init__(fluence, shape, dt_fraction)
        self.mu_rho = np.asarray(mu_rho_by_layer, float)

    def fractions(self, solver):
        tau = np.r_[0.0, np.cumsum(self.mu_rho[solver.layer] * solver.m)]
        return np.exp(-tau[:-1]) - np.exp(-tau[1:])


class UniformDepthSource(PulsedSource):
    """Energy spread uniformly (per unit mass) over the first `depth` metres of layer 0,
    measured from the current surface at the layer's initial density rho0 [kg/m^3].
    Used for the Blanchard & Martin (2005) benchmark load model."""

    def __init__(self, fluence, shape, depth, rho0, dt_fraction=0.1):
        super().__init__(fluence, shape, dt_fraction)
        self.M = depth * rho0

    def fractions(self, solver):
        b = cell_mass_bounds(solver)
        return np.clip(np.minimum(b[1:], self.M) - b[:-1], 0.0, None) / self.M


class VolumetricUniformSource(PulsedSource):
    """Energy spread uniformly per unit volume over the whole domain (e.g. BM2005 neutrons)."""

    def fractions(self, solver):
        dx = solver.dx()
        return dx / dx.sum()


class PeriodicSource(SourceTerm):
    """Repeats a PulsedSource every `period` seconds for `n_shots` shots (shot k starts at k*period)."""

    def __init__(self, base: PulsedSource, period: float, n_shots: int):
        self.base, self.period, self.n = base, period, n_shots

    def _shots(self, t0, t1):
        k0 = max(int(np.floor(t0 / self.period)) - 1, 0)
        k1 = min(int(np.floor(t1 / self.period)), self.n - 1)
        return range(k0, k1 + 1)

    def energy(self, t0, t1):
        return sum(self.base.energy(t0 - k * self.period, t1 - k * self.period) for k in self._shots(t0, t1))

    def fractions(self, solver):
        return self.base.fractions(solver)

    def dt_limit(self, t):
        return min([self.base.dt_limit(t - k * self.period) for k in self._shots(t, t)] + [np.inf])

    def next_event(self, t):
        ev = []
        for k in range(max(int(t // self.period) - 1, 0), min(int(t // self.period) + 2, self.n)):
            ev += [e + k * self.period for e in self.base.shape.events() if e + k * self.period > t * (1 + 1e-14) + 1e-300]
        return min(ev) if ev else np.inf
