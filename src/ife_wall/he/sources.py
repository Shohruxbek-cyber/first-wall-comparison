"""He implantation and displacement-damage sources per shot, from the existing ion spectra.

Uses ife_wall.ions.load_ion_bins (ARIES/NRL 154 MJ tables; number- and energy-conserving bins since
22 Oct 2026, the earlier per-species normalized bins with spectra="normalized"; scaled by Y/154 MJ
and 1/(4 pi R^2)), the time of flight R / v(E) (as in ife_wall.ions), and the SRIM range data /
TRIM-corrected effective projected ranges of ife_wall.srim.

Each tabulated energy bin is split into `n_sub` sub-bins (log spaced in E, number from the
repo's own within-bin spectrum `ions.bin_shape`).  For every sub-bin:
  * He (3He, 4He): implantation profile = gamma distribution with the effective projected
    range and straggling, times the TRIM stopped fraction (1 - backscattered number fraction);
  * all species: NRT (or arc-dpa) displacement profile from damage.DisplacementModel along the
    ion path, blurred by range straggling (Gaussian, mirrored at the surface), times
    (1 - backscattered energy fraction);
  * arrival window [R/v(E_hi), R/v(E_lo)], uniform within the sub-bin.
Time slices (log spaced over the full arrival window) turn this into piecewise-constant rates
G_He(x, slice), G_dpa(x, slice) used by the integrator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr
from scipy.stats import gamma as gamma_dist

from .. import srim
from ..ions import ACCOUNTING_MJ, YIELD_REF, bin_shape, load_ion_bins, speed
from ..scenarios import SCENARIOS
from . import damage as dm

HE_SPECIES = ("4He", "3He")


@dataclass
class SubBin:
    group: str
    species: str
    E: float        # J, number-weighted mean energy of the sub-bin
    E_lo: float
    E_hi: float
    n: float        # ions per m^2 per shot (incident)
    t_lo: float     # arrival window [s]
    t_hi: float


def split_bins(scen, n_sub=6, normalize=True, spectra=None):
    sc = scen.yield_J / YIELD_REF / (4 * np.pi * scen.R ** 2)
    out = []
    for b in load_ion_bins(normalize, spectra=spectra):
        if b.number <= 0:
            continue
        edges = np.geomspace(b.E_lo, b.E_hi, n_sub + 1)
        parts = []
        for a, c in zip(edges[:-1], edges[1:]):
            x = np.geomspace(a, c, 24)
            w = bin_shape(b, x)
            parts.append((np.trapezoid(w, x), np.trapezoid(w * x, x), a, c))
        tot = sum(p[0] for p in parts)
        m = srim.MASS_U[srim.SPECIES_CODE[b.species]]
        for N, EN, a, c in parts:
            if N <= 0:
                continue
            n = b.number * sc * N / tot           # exact: sub-bin numbers sum to the bin number
            out.append(SubBin(b.group, b.species, EN / N, a, c, n,
                              scen.R / float(speed(c, ion_mass(b.species))), scen.R / float(speed(a, ion_mass(b.species)))))
    return out


def ion_mass(species):
    return {"Protons": 1.00728, "Deuterons": 2.01355, "Tritons": 3.01550, "3He": 3.01493, "4He": 4.00151,
            "12C": 11.99671, "Au": 196.96657}[species]


class ShotSources:
    """Per-shot sources on a given set of cell edges."""

    def __init__(self, scen_name, edges, n_sub=6, n_slices=40, stopping=None, dpa_kind="nrt", Ed=None, sub_bins=None,
                 path_element="projected"):
        """path_element: "projected" (production default) counts the displacements of a path
        element over the decrement of the effective projected range; "path" (revision of
        20 Oct 2026) over its path length dE/S (SRIM electronic + nuclear stopping), the count
        that the per-ion TRIM comparison of verify_he_4 uses.  Both place them at the same depth."""
        self.scen = SCENARIOS[scen_name] if scen_name else None
        self.edges = np.asarray(edges, float)
        self.dx = np.diff(self.edges)
        self.x = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.stopping = stopping or srim.SrimStopping(mode="projected", multilayer_trim=True)
        self.sub = list(sub_bins) if sub_bins is not None else split_bins(self.scen, n_sub)
        self.dpa_kind, self.Ed = dpa_kind, Ed
        if path_element not in ("projected", "path"):
            raise ValueError(path_element)
        self.path_element = path_element
        self.he_idx = [i for i, s in enumerate(self.sub) if s.species in HE_SPECIES]
        self._build_profiles()
        self._build_slices(n_slices)

    # ---------------------------------------------------------------- depth profiles
    def _range_m(self, species, E):
        return self.stopping.range(species, "W", E) / dm.RHO_SRIM

    def _strag_m(self, species, E):
        tab = self.stopping.table(species, "W")
        lnE = np.log(tab.E)
        d = np.exp(np.interp(np.log(np.maximum(E, 1e-30)), lnE, np.log(tab.dRp)))
        rp_tab = tab.proj_range(E)
        f = self.stopping.range(species, "W", E) / np.maximum(rp_tab, 1e-300)  # TRIM/table factor
        return d * f / dm.RHO_SRIM

    def he_implant_fraction(self, species, E):
        """Fraction of the incident ions that stop in each cell (gamma depth distribution)."""
        mu = float(self._range_m(species, E))
        sd = float(self._strag_m(species, E))
        k = (mu / sd) ** 2
        th = sd ** 2 / mu
        cdf = gamma_dist.cdf(self.edges, a=k, scale=th)
        stop = 1.0 - self.stopping.reflected_number_fraction(species, "W", E)
        return np.diff(cdf) * stop

    def dpa_per_ion(self, species, E0, model, n_path=70):
        """dpa per ion fluence [dpa per (ion/m^2)] in each cell."""
        E = np.geomspace(E0, 300 * dm.EV, n_path)
        r = self._range_m(species, E)
        s = self._strag_m(species, E)
        mid_E = np.sqrt(E[:-1] * E[1:])
        dl = r[:-1] - r[1:]                                    # path-depth increments (>0)
        mk = 0.5 * (r[0] - r[:-1] + r[0] - r[1:])               # mean depth of each path element
        sk = np.sqrt(np.maximum(s[0] ** 2 - 0.5 * (s[:-1] ** 2 + s[1:] ** 2), 0.0)) + 1e-10
        if self.path_element == "path":                         # path length of each element, dE / S
            S = self.stopping.table(species, "W").stopping(mid_E)   # J per (kg/m^2)
            dl = (E[:-1] - E[1:]) / S / dm.RHO_SRIM
        w = dm.N_AT * model(mid_E) * dl                         # displacements per ion in element
        e = self.edges
        up = (e[None, 1:] - mk[:, None]) / sk[:, None]
        lo = (e[None, :-1] - mk[:, None]) / sk[:, None]
        mir_up = (e[None, 1:] + mk[:, None]) / sk[:, None]
        mir_lo = (e[None, :-1] + mk[:, None]) / sk[:, None]
        frac = (ndtr(up) - ndtr(lo)) + (ndtr(mir_up) - ndtr(mir_lo))
        nd_cell = (w[:, None] * frac).sum(axis=0)               # displacements per ion in each cell
        keep = 1.0 - self.stopping.reflected_energy_fraction(species, "W", E0)
        return keep * nd_cell / (dm.N_AT * self.dx)

    def _build_profiles(self):
        models = {sp: dm.displacement_model(sp, self.dpa_kind, self.Ed) for sp in {s.species for s in self.sub}}
        self.he_prof = np.zeros((len(self.sub), len(self.x)))   # He / m^3 per shot, per sub-bin
        self.dpa_prof = np.zeros((len(self.sub), len(self.x)))  # dpa per shot, per sub-bin
        for i, s in enumerate(self.sub):
            if s.species in HE_SPECIES:
                self.he_prof[i] = s.n * self.he_implant_fraction(s.species, s.E) / self.dx
            self.dpa_prof[i] = s.n * self.dpa_per_ion(s.species, s.E, models[s.species])

    # ---------------------------------------------------------------- time slices
    def _build_slices(self, n_slices):
        t_first = min(s.t_lo for s in self.sub)
        t_last = max(s.t_hi for s in self.sub)
        self.t_edges = np.geomspace(t_first, t_last, n_slices + 1)
        W = np.zeros((len(self.sub), n_slices))
        for i, s in enumerate(self.sub):
            c = np.clip((self.t_edges - s.t_lo) / max(s.t_hi - s.t_lo, 1e-30), 0, 1)
            W[i] = np.diff(c)
        self.slice_frac = W                                    # fraction of sub-bin arriving per slice
        dts = np.diff(self.t_edges)
        self.G_He = (W.T @ self.he_prof) / dts[:, None]        # He / m^3 / s
        self.G_dpa = (W.T @ self.dpa_prof) / dts[:, None]      # dpa / s

    # ---------------------------------------------------------------- summaries
    def he_per_shot_incident(self, group=None):
        return sum(s.n for s in self.sub if s.species in HE_SPECIES and (group is None or s.group == group))

    def he_per_shot_implanted(self, group=None):
        return float(sum((self.he_prof[i] * self.dx).sum() for i in self.he_idx
                         if group is None or self.sub[i].group == group))

    def he_profile(self, group=None):
        idx = [i for i in self.he_idx if group is None or self.sub[i].group == group]
        return self.he_prof[idx].sum(axis=0)

    def dpa_profile(self):
        return self.dpa_prof.sum(axis=0)

    def displacements_per_m2(self):
        return float((self.dpa_profile() * self.dx).sum() * dm.N_AT)
