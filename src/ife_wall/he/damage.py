"""Displacement damage of ions in W: NRT (and arc-dpa) from the primary-knock-on spectrum.

Model (all formulas are standard; sources in Appendix I of the paper):
  * An ion of energy E, species (Z1, A1), moving through W (Z2 = 74, A2 = 183.84) makes W
    recoils (PKAs) of energy T with the Thomas-Fermi differential cross section of Lindhard,
    Scharff & Schiott in the Winterbon-Sigmund-Sanders analytic form
        dsigma = (pi a^2 / 2) f(t^(1/2)) t^(-3/2) dt ,   t = eps^2 T / T_max ,
        f(x) = lam x^(1/3) / [1 + (2 lam x^(4/3))^(2/3)]^(3/2) ,  lam = 1.309 ,
    whose magnitude is rescaled at every E so that  N int T dsigma  equals the SRIM nuclear
    stopping power of the repo's SRIM table (data/srim).  The shape (how much energy goes
    into transfers below the threshold energy) comes from the TF form.
  * Each PKA of energy T gives nu(T_dam(T)) displacements, T_dam from the Lindhard-Robinson
    partition for W in W, nu = NRT (0.8 T_dam / 2 Ed) or arc-dpa (Nordlund 2018).
  * sigma_d(E) = rescale * int dsigma nu   [m^2 per ion].  dpa at depth x for ion fluence F is
    F sigma_d(E(x)), with E(x) from the effective projected-range mapping in ife_wall.srim.
This is a deliberately simple, SPECTER-like calculation; it is checked against TRIM's own
Kinchin-Pease vacancy totals in scripts/verify_he_4_sources.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import srim

Z2, A2 = 74.0, 183.84
ION_Z = {"Protons": 1, "Deuterons": 1, "Tritons": 1, "3He": 2, "4He": 2, "12C": 6, "Au": 79}
N_AT = 6.306e28           # W atoms / m^3 (bcc, 2 / a0^3, a0 = 3.1653 A)
RHO_SRIM = 19350.0        # kg/m^3, density used by SRIM for the range tables
M_W_KG = A2 * 1.66053907e-27
A0_BOHR = 0.529177210903e-10  # m
EV = 1.602176634e-19

# Displacement-model constants (citations in Appendix I of the paper)
ED_NRT = 90.0             # eV, ASTM E521 value for W
ARC = dict(Ed=70.0, b=-0.56, c=0.12)  # Nordlund et al. 2018, W (value as tabulated there)


def mass_u(species):
    return srim.MASS_U[srim.SPECIES_CODE[species]]


def E_L(Z1, A1):
    """Lindhard unit of energy [eV] for the ion-W pair."""
    return 30.724 * Z1 * Z2 * np.sqrt(Z1 ** (2 / 3) + Z2 ** (2 / 3)) * (A1 + A2) / A2


def tdam_lindhard(T_eV):
    """Lindhard-Robinson damage energy [eV] of a W recoil of energy T [eV] in W."""
    T = np.asarray(T_eV, float)
    k = 0.1337 * Z2 ** (1 / 6) * np.sqrt(Z2 / A2)
    eps = T / E_L(Z2, A2)
    g = 3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps
    return T / (1.0 + k * g)


def nu_nrt(Td, Ed=ED_NRT):
    Td = np.asarray(Td, float)
    return np.where(Td < Ed, 0.0, np.where(Td < 2 * Ed / 0.8, 1.0, 0.8 * Td / (2 * Ed)))


def nu_arc(Td, Ed=ARC["Ed"], b=ARC["b"], c=ARC["c"]):
    Td = np.asarray(Td, float)
    xi = (1 - c) * (np.maximum(Td, 1e-30) / (2 * Ed / 0.8)) ** b + c
    return np.where(Td < Ed, 0.0, np.where(Td < 2 * Ed / 0.8, 1.0, 0.8 * Td / (2 * Ed) * xi))


def _f_tf(t, lam=1.309):
    x = np.sqrt(t)
    return lam * x ** (1 / 3) / (1.0 + (2 * lam * x ** (4 / 3)) ** (2 / 3)) ** 1.5


def _sn_srim(species, E_J):
    """SRIM nuclear stopping cross section per W atom [J m^2] at ion energy E [J]."""
    tab = srim.SrimTable(srim.SPECIES_CODE[species], "W")
    lnE = np.log(tab.E)
    sn = np.exp(np.interp(np.log(E_J), lnE, np.log(tab.Sn)))
    return sn * M_W_KG  # J m^2/kg * kg/atom


class DisplacementModel:
    """sigma_d(E) [m^2/ion] on an energy grid, for one species and one displacement model."""

    def __init__(self, species, nu_fn=nu_nrt, Ed_nu=ED_NRT, n_E=140, n_T=900):
        self.species = species
        Z1, A1 = ION_Z[species], mass_u(species)
        gam = 4 * A1 * A2 / (A1 + A2) ** 2
        aL = 0.8853 * A0_BOHR / np.sqrt(Z1 ** (2 / 3) + Z2 ** (2 / 3))
        EL = E_L(Z1, A1)
        self.E_J = np.geomspace(200.0, 3e7, n_E) * EV    # 200 eV .. 30 MeV
        sig, sig_n_tf = [], []
        for E in self.E_J / EV:
            Tm = gam * E
            eps = E / EL
            t = np.geomspace(1e-14, eps ** 2, n_T)
            T = Tm * t / eps ** 2
            # d sigma = (pi a^2/2) f t^-3/2 dt = (pi a^2/2) f t^-1/2 dlnT
            ds = (np.pi * aL ** 2 / 2) * _f_tf(t) * t ** -0.5            # dsigma per d ln T
            sn_tf = np.trapezoid(ds * T, np.log(T)) * 1.0                # eV m^2 (energy-weighted)
            Td = tdam_lindhard(T)
            nd = np.trapezoid(ds * nu_fn(Td), np.log(T)) if Tm > Ed_nu else 0.0
            sig.append(nd)
            sig_n_tf.append(sn_tf * EV)                                   # J m^2
        sig, sig_n_tf = np.array(sig), np.array(sig_n_tf)
        sn_srim = _sn_srim(species, self.E_J)
        self.rescale = sn_srim / sig_n_tf
        self.sigma_d = sig * self.rescale
        self._lnE = np.log(self.E_J)

    def __call__(self, E_J):
        E = np.asarray(E_J, float)
        return np.interp(np.log(np.maximum(E, 1e-30)), self._lnE, self.sigma_d, left=0.0, right=self.sigma_d[-1])


@lru_cache(maxsize=None)
def displacement_model(species, kind="nrt", Ed=None):
    if kind == "nrt":
        Ed = ED_NRT if Ed is None else Ed
        return DisplacementModel(species, lambda Td, Ed=Ed: nu_nrt(Td, Ed), Ed)
    if kind == "arc":
        return DisplacementModel(species, nu_arc, ARC["Ed"])
    raise ValueError(kind)
