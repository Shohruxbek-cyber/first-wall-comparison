"""Material models for the 1D thermal solver.

All quantities SI: T [K], rho [kg/m^3], k [W/(m K)], cp [J/(kg K)], h [J/kg],
p_sat [Pa], L_v [J/kg], molar mass [kg/mol].

A Material stores a tabulated specific enthalpy h(T) (sensible heat integrated from cp,
plus the latent heat of fusion spread over a narrow "mushy" interval dTm around Tm).
The solver works in enthalpy, so T(h) and dT/dh come from the same table and melting
is handled without special cases.

Production property correlations (W, Li, substrate) are NOT defined here; they are in
library.py, with their sources (Table B.1 of the paper).  `constant_material` is for
verification tests only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

R_GAS = 8.314462618  # J/(mol K), CODATA 2018
T_REF = 298.15  # K, enthalpy reference temperature

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Vapor:
    """Vapor-side properties for Hertz-Knudsen-Langmuir evaporation."""

    p_sat: ArrayFn  # Pa, function of T [K]
    L_v: ArrayFn  # J/kg, latent heat of vaporization at T [K]
    molar_mass: float  # kg/mol
    source: str = ""

    def hkl_flux(self, T: np.ndarray, alpha: float) -> np.ndarray:
        """Net evaporative mass flux, kg/(m^2 s): j = alpha p_sat(T) sqrt(M / (2 pi R T))."""
        T = np.asarray(T, dtype=float)
        return alpha * self.p_sat(T) * np.sqrt(self.molar_mass / (2.0 * np.pi * R_GAS * T))

    def hkl_flux_dT(self, T: float, alpha: float, dT: float = 1e-3) -> float:
        """Centered finite-difference derivative dj/dT, kg/(m^2 s K)."""
        return float((self.hkl_flux(T + dT, alpha) - self.hkl_flux(T - dT, alpha)) / (2 * dT))


@dataclass
class Material:
    name: str
    rho: ArrayFn  # kg/m^3
    k: ArrayFn  # W/(m K)
    cp: ArrayFn  # J/(kg K); may be discontinuous at Tm
    T_min: float = 200.0
    T_max: float = 8000.0
    Tm: Optional[float] = None  # melting point, K (None: no phase change modelled)
    L_f: float = 0.0  # latent heat of fusion, J/kg
    dTm: float = 1.0  # width of the mushy interval, K
    vapor: Optional[Vapor] = None
    sources: dict = field(default_factory=dict)  # property -> citation (for provenance)
    dT_grid: float = 0.5  # K, base spacing of the enthalpy table

    def __post_init__(self):
        grid = np.arange(self.T_min, self.T_max + self.dT_grid, self.dT_grid)
        if self.Tm is not None:
            lo, hi = self.Tm - self.dTm / 2, self.Tm + self.dTm / 2
            grid = np.union1d(grid, np.linspace(lo, hi, 21))
            grid = np.union1d(grid, [lo - 1e-9, hi + 1e-9])
        # Trapezoidal integral of cp; cp is sampled on the midpoints so a jump at Tm is
        # captured to within one grid interval.
        mid = 0.5 * (grid[1:] + grid[:-1])
        dh = self.cp(mid) * np.diff(grid)
        h = np.concatenate([[0.0], np.cumsum(dh)])
        if self.Tm is not None and self.L_f > 0:
            frac = np.clip((grid - (self.Tm - self.dTm / 2)) / self.dTm, 0.0, 1.0)
            h = h + self.L_f * frac
        h -= np.interp(T_REF, grid, h) if grid[0] <= T_REF else h[0]
        if np.any(np.diff(h) <= 0):
            raise ValueError(f"{self.name}: enthalpy table not strictly increasing")
        self._T, self._h = grid, h
        self._dTdh = np.diff(grid) / np.diff(h)  # per segment
        # Tabulate rho(T) and k(T) on the same grid (piecewise-linear interpolation is exact to
        # << 0.1 % on a 0.5 K grid; a jump at Tm is resolved within the mushy interval).  The
        # original correlations stay available as rho_fn / k_fn.
        self.rho_fn, self.k_fn = self.rho, self.k
        rho_tab, k_tab = np.asarray(self.rho_fn(grid), float), np.asarray(self.k_fn(grid), float)
        self.rho = lambda T, _g=grid, _v=rho_tab: np.interp(T, _g, _v)
        self.k = lambda T, _g=grid, _v=k_tab: np.interp(T, _g, _v)

    # --- enthalpy table --------------------------------------------------------------
    def h_of_T(self, T):
        self._check(T)
        return np.interp(T, self._T, self._h)

    def T_of_h(self, h):
        h = np.asarray(h, dtype=float)
        if np.any(h < self._h[0]) or np.any(h > self._h[-1]):
            raise ValueError(f"{self.name}: enthalpy outside table ({self.T_min}-{self.T_max} K)")
        return np.interp(h, self._h, self._T)

    def dTdh(self, h):
        """Slope dT/dh of the segment containing h (piecewise-constant)."""
        i = np.clip(np.searchsorted(self._h, h, side="right") - 1, 0, len(self._dTdh) - 1)
        return self._dTdh[i]

    def liquid_fraction(self, h):
        if self.Tm is None:
            return np.zeros_like(np.asarray(h, dtype=float))
        h_sol = np.interp(self.Tm - self.dTm / 2, self._T, self._h)
        h_liq = np.interp(self.Tm + self.dTm / 2, self._T, self._h)
        return np.clip((np.asarray(h) - h_sol) / (h_liq - h_sol), 0.0, 1.0)

    def _check(self, T):
        T = np.asarray(T)
        if np.any(T < self.T_min) or np.any(T > self.T_max):
            raise ValueError(f"{self.name}: T outside property range {self.T_min}-{self.T_max} K")


def constant_material(name, rho, k, cp, Tm=None, L_f=0.0, dTm=1.0, k_liq=None, cp_liq=None,
                      vapor=None, T_min=1.0, T_max=20000.0, dT_grid=0.5):
    """Constant-property material (optionally different liquid k and cp). VERIFICATION ONLY."""

    def piecewise(solid, liquid):
        if liquid is None or Tm is None:
            return lambda T: np.full_like(np.asarray(T, dtype=float), solid)
        return lambda T: np.where(np.asarray(T) >= Tm, liquid, solid).astype(float)

    return Material(name=name, rho=lambda T: np.full_like(np.asarray(T, dtype=float), rho),
                    k=piecewise(k, k_liq), cp=piecewise(cp, cp_liq), Tm=Tm, L_f=L_f, dTm=dTm,
                    vapor=vapor, T_min=T_min, T_max=T_max, dT_grid=dT_grid,
                    sources={"all": "test-only constant properties"})
