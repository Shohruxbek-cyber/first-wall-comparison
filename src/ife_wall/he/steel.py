"""Helium, hydrogen-isotope and displacement sources in the steel behind a lithium film (revision of
21 Oct 2026).  New code; nothing else in ife_wall is changed by it.

The incident ions are those of ife_wall.he.sources.split_bins (ARIES/NRL 154 MJ spectra in the production
construction of ife_wall.ions, scaled by Y/154 MJ and 1/(4 pi R^2)).  Each sub-bin crosses the film with the production
range-energy mapping (srim.SrimStopping, projected mode, as in the deposition of the lithium runs): an
ion of energy E reaches the steel with E_res = E(R_Li(E) - sigma_film) if R_Li(E) > sigma_film, where
sigma_film is the areal mass of the film at its starting temperature (density of library.lithium).
In the steel (modelled as iron: SRIM Fe tables, Z2 = 26, A2 = 55.847, SRIM default density) every
transmitted helium and hydrogen ion is counted as implanted (no TRIM data for Fe, so no reflection),
and the displacements follow the NRT model of ife_wall.he.damage with the iron constants, along the
projected-range mapping in iron, blurred by the SRIM range straggling (Gaussian, mirrored at the
interface), exactly as ShotSources.dpa_per_ion does in tungsten.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from .. import srim
from ..library import lithium
from ..scenarios import SCENARIOS
from . import damage as dm
from .sources import HE_SPECIES, split_bins

H_SPECIES = ("Protons", "Deuterons", "Tritons")
AMU = 1.66053907e-27
N_AVOGADRO = 6.02214076e23

# iron as the steel (SRIM default target of data/srim/*_in_Fe)
FE_Z2, FE_A2 = 26.0, 55.847
FE_RHO_SRIM = 7865.8                                  # kg/m^3, SRIM default density of Fe
FE_N_AT = FE_RHO_SRIM / (FE_A2 * 1e-3) * N_AVOGADRO   # atoms/m^3
FE_ED_NRT = 40.0                                      # eV, NRT threshold for Fe (Nordlund et al. 2018, Table 1)


def _E_L(Z1, A1, Z2, A2):
    return 30.724 * Z1 * Z2 * np.sqrt(Z1 ** (2 / 3) + Z2 ** (2 / 3)) * (A1 + A2) / A2


def _tdam(T_eV, Z2, A2):
    T = np.asarray(T_eV, float)
    k = 0.1337 * Z2 ** (1 / 6) * np.sqrt(Z2 / A2)
    eps = T / _E_L(Z2, A2, Z2, A2)
    g = 3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps
    return T / (1.0 + k * g)


class TargetDisplacementModel:
    """damage.DisplacementModel for any SRIM target: same Thomas-Fermi cross section rescaled to the SRIM
    nuclear stopping of that target, same Lindhard-Robinson partition, NRT displacements with threshold Ed.
    With the tungsten constants it reproduces damage.DisplacementModel (tests/test_steel.py)."""

    def __init__(self, species, target, Z2, A2, Ed, n_E=140, n_T=900):
        Z1, A1 = dm.ION_Z[species], dm.mass_u(species)
        gam = 4 * A1 * A2 / (A1 + A2) ** 2
        aL = 0.8853 * dm.A0_BOHR / np.sqrt(Z1 ** (2 / 3) + Z2 ** (2 / 3))
        EL = _E_L(Z1, A1, Z2, A2)
        self.E_J = np.geomspace(200.0, 3e7, n_E) * dm.EV
        sig, sig_n_tf = [], []
        for E in self.E_J / dm.EV:
            Tm = gam * E
            eps = E / EL
            t = np.geomspace(1e-14, eps ** 2, n_T)
            T = Tm * t / eps ** 2
            ds = (np.pi * aL ** 2 / 2) * dm._f_tf(t) * t ** -0.5
            sn_tf = np.trapezoid(ds * T, np.log(T)) * 1.0
            Td = _tdam(T, Z2, A2)
            nd = np.trapezoid(ds * dm.nu_nrt(Td, Ed), np.log(T)) if Tm > Ed else 0.0
            sig.append(nd)
            sig_n_tf.append(sn_tf * dm.EV)
        sig, sig_n_tf = np.array(sig), np.array(sig_n_tf)
        tab = srim.SrimTable(srim.SPECIES_CODE[species], target)
        lnE = np.log(tab.E)
        sn = np.exp(np.interp(np.log(self.E_J), lnE, np.log(tab.Sn))) * A2 * AMU   # J m^2 per target atom
        self.sigma_d = sig * sn / sig_n_tf
        self._lnE = np.log(self.E_J)

    def __call__(self, E_J):
        E = np.asarray(E_J, float)
        return np.interp(np.log(np.maximum(E, 1e-30)), self._lnE, self.sigma_d, left=0.0, right=self.sigma_d[-1])


@lru_cache(maxsize=None)
def fe_model(species):
    return TargetDisplacementModel(species, "Fe", FE_Z2, FE_A2, FE_ED_NRT)


class SteelBehindFilm:
    """Per-shot helium, hydrogen and displacement sources in the steel behind a lithium film."""

    def __init__(self, scen_name, delta_m, T_li=600.0, n_sub=24, depth_m=1.2e-4, n_cells=2400, sub_bins=None):
        self.scen = SCENARIOS[scen_name]
        self.delta = float(delta_m)
        self.sigma_film = float(lithium().rho(np.array(T_li))) * self.delta     # kg/m^2
        self.stopping = srim.SrimStopping(mode="projected")
        self.sub = list(sub_bins) if sub_bins is not None else split_bins(self.scen, n_sub)
        self.edges = np.linspace(0.0, depth_m, n_cells + 1)
        self.dx = np.diff(self.edges)
        self.x = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.E_res = np.array([self._residual(s.species, s.E) for s in self.sub])

    def _residual(self, species, E):
        R0 = float(self.stopping.range(species, "Li", E))
        if R0 <= self.sigma_film:
            return 0.0
        return float(self.stopping.energy_at_range(species, "Li", R0 - self.sigma_film))

    # ---------------------------------------------------------------- fluences
    def fluence(self, species_set, group=None):
        """ions / m^2 per shot reaching the steel."""
        return float(sum(s.n for s, E in zip(self.sub, self.E_res)
                         if E > 0 and s.species in species_set and (group is None or s.group == group)))

    def energy_into_steel(self, group=None):
        """J / m^2 per shot carried into the steel by the ions."""
        return float(sum(s.n * E for s, E in zip(self.sub, self.E_res) if E > 0 and (group is None or s.group == group)))

    # ---------------------------------------------------------------- displacements
    def _range_fe(self, species, E):
        return self.stopping.range(species, "Fe", E) / FE_RHO_SRIM

    def _strag_fe(self, species, E):
        tab = self.stopping.table(species, "Fe")
        lnE = np.log(tab.E)
        return np.exp(np.interp(np.log(np.maximum(E, 1e-30)), lnE, np.log(tab.dRp))) / FE_RHO_SRIM

    def _dpa_per_ion(self, species, E0, n_path=70):
        E = np.geomspace(E0, 300 * dm.EV, n_path)
        r = self._range_fe(species, E)
        s = self._strag_fe(species, E)
        mid_E = np.sqrt(E[:-1] * E[1:])
        dl = r[:-1] - r[1:]
        mk = 0.5 * (r[0] - r[:-1] + r[0] - r[1:])
        sk = np.sqrt(np.maximum(s[0] ** 2 - 0.5 * (s[:-1] ** 2 + s[1:] ** 2), 0.0)) + 1e-10
        w = FE_N_AT * fe_model(species)(mid_E) * dl
        e = self.edges
        frac = (ndtr((e[None, 1:] - mk[:, None]) / sk[:, None]) - ndtr((e[None, :-1] - mk[:, None]) / sk[:, None])
                + ndtr((e[None, 1:] + mk[:, None]) / sk[:, None]) - ndtr((e[None, :-1] + mk[:, None]) / sk[:, None]))
        return (w[:, None] * frac).sum(axis=0) / (FE_N_AT * self.dx)

    def dpa_profile(self, group=None):
        """dpa per shot in each steel cell (all transmitted species)."""
        out = np.zeros(len(self.x))
        for s, E in zip(self.sub, self.E_res):
            if E > 0 and (group is None or s.group == group):
                out += s.n * self._dpa_per_ion(s.species, E)
        return out

    def summary(self):
        prof = self.dpa_profile()
        five = self.x < 5e-6
        he = self.fluence(HE_SPECIES)
        return {
            "delta_um": self.delta * 1e6, "film_areal_mass_kg_m2": self.sigma_film,
            "He_per_m2_per_shot": {"burn": self.fluence(HE_SPECIES, "burn"), "debris": self.fluence(HE_SPECIES, "debris"),
                                   "total": he},
            "H_isotopes_per_m2_per_shot": {"burn": self.fluence(H_SPECIES, "burn"), "debris": self.fluence(H_SPECIES, "debris"),
                                           "total": self.fluence(H_SPECIES)},
            "ion_energy_into_steel_J_m2": self.energy_into_steel(),
            "dpa_per_shot_peak": float(prof.max()) if prof.any() else 0.0,
            "dpa_peak_depth_um": float(self.x[int(np.argmax(prof))] * 1e6) if prof.any() else None,
            "dpa_per_shot_mean_0_5um": float((prof[five] * self.dx[five]).sum() / 5e-6),
            "shots_to_1e18_He_m2": 1e18 / he if he > 0 else None,
            "shots_to_1e19_He_m2": 1e19 / he if he > 0 else None,
        }
