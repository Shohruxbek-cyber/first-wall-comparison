"""Physical constants of the He-retention model, with their sources (full table: Table I.3 of the paper).

Every value below comes from the literature (Table I.3 of the paper lists them with their
sources).  Nothing here is tuned to a result.  Where sources disagree a baseline and a
low/high range are given (RANGES), and the range is run as a sensitivity case.  Values that could
not be found are listed in MISSING and are NOT used.

Short source keys: Faney2013 = T. Faney, PhD thesis, UC Berkeley (2013); He2025 = He et al.,
arXiv:2512.18258; Tirumala2025 = Tirumala et al., arXiv:2509.22012 (DFT, preprint);
Perez2014 = Perez, Vogel, Uberuaga, PRB 90, 014102; Mohamed2025 = Mohamed et al.,
arXiv:2407.06022; Nordlund2018 = Nat. Commun. 9, 1084.
"""
from __future__ import annotations

import numpy as np

from .model import Params

A0 = 3.1653e-10          # m, W lattice parameter (a0 = 0.3165 nm; Mohamed2025 0.316 nm; N_W = 2/a0^3)
N_W = 6.306e28           # W atoms / m^3   (2 / a0^3 for a0 = 3.1653 A; DM2021 quote 6.3e28)

# --- migration: D = D0 exp(-Em/kT)
D0_HE, EM_HE = 2.95e-8, 0.13     # m^2/s, eV   Faney2013 Table 3.1 (MD, 300-2000 K Arrhenius fit)
D0_V, EM_V = 1.11e-6, 1.71       # He2025 Table 2 (mixed experiment/DFT/potential)
D0_I, EM_I = 8.77e-8, 0.009      # Faney2013 Table 3.2 (MD, I_1); He2025 Table 2 lists the same numbers

# --- He_k V binding energies (eV): binding of the k-th He to He_(k-1) V
EB_BASE = np.array([4.05, 3.11, 2.88, 2.41,   # k = 1..4: Kornelsen TDS via Faney2013 Table 3.5 / 5.1
                    2.21, 2.27, 2.45])        # k = 5..7: DFT, Tirumala2025 Table D.1 (preprint; derived by us)
EB_LOW = np.array([4.05, 2.89, 2.52, 2.41, 2.0, 2.0, 2.0])   # min over sources (k = 5-7: rounded bound, not data)
EB_HIGH = np.array([4.78, 3.17, 3.14, 3.19, 2.7, 2.7, 2.7])  # max over sources (k = 5-7: rounded bound, not data)
EB_FLAT = np.array([4.05, 3.11, 2.88, 2.41, 2.4, 2.4, 2.4])  # Faney's suggestion: flat 2.4 eV beyond k = 4
K_MAX = 7   # largest k with a published binding energy (He7V); trap mutation sets in at He_5..He_7 (Level 2)

# --- radii (diffusion-limited reaction rate k = 4 pi (rA + rB)(DA + DB), Faney2013 Eq. 2.6)
R_HE1, R_V1 = 0.30e-9, (np.sqrt(3.0) / 4.0) * A0          # Faney2013 Sec. 3.5
R_C = R_HE1 + R_V1                                        # He + V capture radius = 0.437 nm
R_IV = R_V1 + 1.15 * R_V1                                 # V + I: 0.295 nm (Faney2013 Eqs. 3.13-3.15)
R_IV_HIGH = 0.465e-9                                      # unverified snippet value (item 5f), sensitivity only

NU = 1e13                                                 # s^-1  Dunand2023; Faney2013 (implied 1.0e13)
NU_LOW, NU_HIGH = 1e12, 1.3e13

# --- sinks (loss rate = D * S * C)
RHO_D = 5e12                                              # m^-2, Mohamed2025 Table 2 (from Shah2020)
Z_I, Z_V = 1.2, 1.0                                       # dislocation bias for SIA / vacancy (Mohamed2025 Table 2)
RHO_D_LOW, RHO_D_HIGH = 2e10, 1e14                        # annealed (Mason2021) / as-rolled (unverified snippet)

# --- displacement damage
ED_NRT = 90.0                                             # eV, ASTM E521 (as quoted by He2025, Wielunska2020)
ARC_ED, ARC_B, ARC_C = 70.0, -0.56, 0.12                  # Nordlund2018 Table 1, W row
XI_ARC_10KEV = (1 - ARC_C) * (10e3 / (2 * ARC_ED / 0.8)) ** ARC_B + ARC_C   # = 0.211: arc-dpa efficiency at T_dam = 10 keV
XI_BASE = 1.0                                             # NRT dpa = Frenkel pairs (baseline); low case xi = XI_ARC_10KEV

# --- neutrons (minor; Sharafat2005 Sec. 3.2, Gilbert2013/Stork2014)
NEUTRON_DPA_PER_SHOT = 2e-7        # 1e-6 dpa/s at 5 Hz, HAPL 154 MJ at 6.5 m (Sharafat2005; material not stated)
NEUTRON_HE_PER_DPA = 0.3           # appm He per dpa, W DEMO first wall (derived from Gilbert2013, Stork2014); 0.14-0.9

# --- Level 2 (He self-trapping and trap mutation): sensitivity case only
L2_EBC = np.array([1.00, 1.35, 1.69, 1.84, 1.84, 1.84])   # binding of 2nd..7th He to He_(m-1) (Tirumala2025 DFT, derived); m=6,7 held flat (MISSING)
L2_NU = 1e13                                             # He_N breakup prefactor 1e12-1e13 (Perez2014 Table II)
L2_NUTM = np.array([0, 0, 0, 2.95e12, 1.28e13, 6.64e12])  # trap mutation prefactors, N = 2..7 (Perez2014 Table III, N = 5, 6, 7)
L2_ETM = np.array([0, 0, 0, 1.20, 1.02, 0.701])           # eV

MISSING = [
    "D0 for DFT (0.06 eV) and experimental (0.24-0.32 eV) He migration energies",
    "He_8 V binding energy (no source); k = 5-7 from one DFT preprint",
    "Becquart & Domain He_n V binding energies as numbers (plot only in Faney2013)",
    "V-I recombination radius from a primary source (derived from Faney2013 radii)",
    "He binding to dislocations / grain boundaries",
    "He-He cluster binding for m >= 6 (held flat at the m = 5 value)",
]


def level2_params(r_he1=R_HE1):
    """Level-2 dictionary for Params.level2 (He_2 .. He_7 clusters, trap mutation)."""
    J = len(L2_EBC)
    om = (3.0 / (4 * np.pi)) * A0 ** 3 / 10.0           # He volume in a He cluster (Faney2013 Eq. 3.14)
    sizes = np.arange(2, J + 2)
    r_cl = r_he1 + (om * sizes) ** (1 / 3) - om ** (1 / 3)
    return dict(J=J, EbC=L2_EBC.copy(), rCl=r_he1 + r_cl, nutm=L2_NUTM.copy(), Etm=L2_ETM.copy(),
                nuHeHe=L2_NU, rHe1=r_he1)


def baseline(**kw):
    """Baseline Level-1 parameter set.  Keyword overrides replace individual Params fields."""
    d = dict(K=K_MAX, D0_He=D0_HE, Em_He=EM_HE, D0_V=D0_V, Em_V=EM_V, D0_I=D0_I, Em_I=EM_I,
             r_c=R_C, r_iv=R_IV, nu=NU, Eb=EB_BASE.copy(), sink_He=0.0, sink_V=Z_V * RHO_D, sink_I=Z_I * RHO_D,
             surface_factor=1.0, ann_hev=True, level2=None)
    d.update(kw)
    return Params(**d)
