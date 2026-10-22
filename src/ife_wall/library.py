"""Production material models.  The source of every correlation is given below and in Table F.1 of the paper.

T in K unless stated.  Short names:
  Tolias2017   arXiv:1703.06302v4, recommended set p. 18
  Raffray2005  J. Nucl. Mater. 347:178, Table 2 (W vapour pressure, heat of vaporization)
  Davison1968  NASA TN D-4650 (liquid Li), printed pages
  BM2005       Blanchard & Martin, J. Nucl. Mater. 347:192, Appendix A (steel; ITER-MPH W)
"""
from __future__ import annotations

import numpy as np

from .materials import Material, Vapor

M_W = 0.18384  # kg/mol (IUPAC)
M_LI = 0.00694  # kg/mol (IUPAC)
LN10 = np.log(10.0)
R_GAS = 8.314462618

# ------------------------------------------------------------------------------------------
# Tungsten, baseline (Tolias2017 + Raffray2005 vapour)
# ------------------------------------------------------------------------------------------
W_TM = 3695.0  # K, Tolias2017 p. 2
W_T0 = 293.15  # K, Tolias2017 pp. 10, 18
W_LF = 52.3e3 / M_W  # J/kg (52.3 kJ/mol, Tolias2017 p. 18) = 284.5 kJ/kg

# Heat of vaporization options (J/kg)
W_LV = {
    "raffray_table": 4009e3,  # Raffray2005 Table 2 (SOURCED, secondary)
    "raffray_slope": 44485 * LN10 * R_GAS / M_W,  # Clausius-Clapeyron slope of the same p_sat law (DERIVED, 4.63 MJ/kg)
}


def _w_rho(T):
    T = np.asarray(T, float)
    s = 19.25 - 2.66207e-4 * (T - W_T0) - 3.0595e-9 * (T - W_T0) ** 2 - 9.5185e-12 * (T - W_T0) ** 3
    l = 16.267 - 7.679e-4 * (T - W_TM) - 8.091e-8 * (T - W_TM) ** 2
    return 1e3 * np.where(T < W_TM, s, l)  # g/cm^3 -> kg/m^3


def _w_cp(T):
    T = np.asarray(T, float)
    c1 = 21.868372 + 8.068661e-3 * T - 3.756196e-6 * T ** 2 + 1.075862e-9 * T ** 3 + 1.406637e4 / T ** 2
    c2 = 2.022 + 1.315e-2 * T
    cl = np.full_like(T, 51.3)
    return np.select([T < 3080.0, T < W_TM], [c1, c2], cl) / M_W  # J/(mol K) -> J/(kg K)


def _w_k(T):
    T = np.asarray(T, float)
    s = 149.441 - 45.466e-3 * T + 13.193e-6 * T ** 2 - 1.484e-9 * T ** 3 + 3.866e6 / T ** 2
    l = 66.6212 + 0.02086 * (T - W_TM) - 3.7585e-6 * (T - W_TM) ** 2
    return np.where(T < W_TM, s, l)


def w_vapor(lv_option="raffray_table"):
    L = W_LV[lv_option]
    return Vapor(p_sat=lambda T: 10.0 ** (12.74 - 44485.0 / np.asarray(T, float)),
                 L_v=lambda T: np.full_like(np.asarray(T, float), L), molar_mass=M_W,
                 source=f"p_sat: Raffray2005 Table 2; L_v: {lv_option}")


def tungsten(lv_option="raffray_table", T_max=8000.0):
    """Tolias2017 fits.  Liquid fits are stated for 3695-6000 K; above 6000 K they are
    extrapolated (any run exceeding 6000 K is flagged in its output)."""
    return Material("W", rho=_w_rho, k=_w_k, cp=_w_cp, T_min=250.0, T_max=T_max, Tm=W_TM, L_f=W_LF,
                    dTm=1.0, vapor=w_vapor(lv_option),
                    sources={"rho,cp,k,Tm,Lf": "Tolias2017 p.18", "vapor": "Raffray2005 Table 2"})


# ------------------------------------------------------------------------------------------
# Tungsten as used by Blanchard & Martin (ITER MPH polynomials, T in deg C) -- validation only
# ------------------------------------------------------------------------------------------
def tungsten_bm2005():
    C = lambda T: np.asarray(T, float) - 273.15
    return Material("W (BM2005 App. A)",
                    rho=lambda T: 19302.7 - 0.23786 * C(T) - 2.2448e-5 * C(T) ** 2,
                    cp=lambda T: 128.3 + 0.0328 * C(T) - 3.41e-6 * C(T) ** 2,
                    k=lambda T: 174.9 - 0.107 * C(T) + 5.01e-5 * C(T) ** 2 - 7.835e-9 * C(T) ** 3,
                    T_min=250.0, T_max=3695.0, sources={"all": "BM2005 Appendix A (ITER MPH)"})


# ------------------------------------------------------------------------------------------
# Liquid lithium (Davison1968)
# ------------------------------------------------------------------------------------------
LI_TM = 453.7  # K, Davison1968 p. 4
LI_LV = {
    "davison_slope": 8064.5 * LN10 * R_GAS / M_LI,  # ESTIMATED: Clausius-Clapeyron slope of Eq. (9), 22.25 MJ/kg
    "handbook_147kJmol": 147e3 / M_LI,  # ESTIMATED: author-supplied handbook value ~147 kJ/mol (citation pending), 21.2 MJ/kg
}


def li_vapor(lv_option="davison_slope"):
    L = LI_LV[lv_option]
    return Vapor(p_sat=lambda T: 10.0 ** (10.015 - 8064.5 / np.asarray(T, float)),
                 L_v=lambda T: np.full_like(np.asarray(T, float), L), molar_mass=M_LI,
                 source=f"p_sat: Davison1968 Eq. (9); L_v: {lv_option} (ESTIMATED)")


def lithium(lv_option="davison_slope", T_max=4000.0):
    """Liquid Li only (no phase change).  Davison fits are for T_m to ~1600 K; above that
    they are extrapolated (flagged in outputs).  Li critical temperature is well above
    T_max, but p_sat at thousands of K is far outside the data (800-1800 K)."""
    return Material("Li(l)", rho=lambda T: 562.0 - 0.100 * np.asarray(T, float),
                    cp=lambda T: np.full_like(np.asarray(T, float), 4169.0),
                    k=lambda T: 21.874 + 0.056255 * np.asarray(T, float) - 1.8325e-5 * np.asarray(T, float) ** 2,
                    T_min=LI_TM, T_max=T_max, vapor=li_vapor(lv_option),
                    sources={"rho": "Davison1968 Eq.(1)", "cp": "Eq.(5)", "k": "p.11 polynomial", "p_sat": "Eq.(9)"})


# ------------------------------------------------------------------------------------------
# Ferritic steel substrate (BM2005 App. A; labelled substitute for ODS-RAFM)
# ------------------------------------------------------------------------------------------
def ferritic_steel(T_max=1800.0, k=33.0):
    """k: constant conductivity, W/(m K); 33 is the BM2005 value (default); other values only for the
    steel-conductivity sensitivity of 21 Oct 2026 (Options.steel_k)."""
    return Material("ferritic steel (BM2005; substitute for ODS-RAFM)",
                    rho=lambda T: np.full_like(np.asarray(T, float), 7800.0),
                    cp=lambda T: 500.0 + 0.6 * (np.asarray(T, float) - 273.15),
                    k=lambda T: np.full_like(np.asarray(T, float), k),
                    T_min=250.0, T_max=T_max, sources={"all": "BM2005 Appendix A (Klueh 2000; Shiba 1996)"})
