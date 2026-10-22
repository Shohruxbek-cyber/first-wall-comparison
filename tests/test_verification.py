"""Verification tests.  Must pass before any production run.

Every material in this file is a constant-property TEST material; the W-like numbers are
chosen only to give realistic length/time scales and are not used anywhere else.
Each test prints its measured error; run `pytest -s` to see them.  The numbers are also
written to runs/verification/results.json.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, erfc, erfcx

from ife_wall.ions import MonoIonSource, StoppingTable, MEV
from ife_wall.materials import Vapor, constant_material
from ife_wall.source_term import BeerLawSource, SquarePulse
from ife_wall.thermal import BackBC, FrontBC, Layer, Mesh, ThermalSolver

RESULTS = {}
OUT = Path(__file__).resolve().parents[1] / "runs" / "verification"

RHO, K, CP = 19300.0, 100.0, 140.0  # test-only, W-like magnitudes
ALPHA = K / (RHO * CP)


def report(name, **vals):
    RESULTS[name] = {k: float(v) if np.ndim(v) == 0 else [float(x) for x in v] for k, v in vals.items()}
    print(f"\n[{name}] " + ", ".join(f"{k}={v}" for k, v in RESULTS[name].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))


def closure(s):
    return abs(s.ledger.residual()[1])


# ---------------------------------------------------------------------------------------
# 1. Semi-infinite slab, constant surface flux
# ---------------------------------------------------------------------------------------
def test_1_semi_infinite_constant_flux():
    mat = constant_material("test", RHO, K, CP)
    q = 1e9  # W/m^2
    errs = []
    for t_end in (1e-8, 1e-7, 1e-6, 1e-5):
        L = 20 * np.sqrt(ALPHA * t_end)
        s = ThermalSolver(Mesh.build([Layer(mat, L, L / 2e4, 1.1)], 300.0), 300.0,
                          front=FrontBC("flux", flux=lambda t: q))
        s.run(t_end, dt0=t_end * 1e-6, dT_target=1.0)
        exact = 2 * q / K * np.sqrt(ALPHA * t_end / np.pi)
        errs.append((s.surface_temperature() - 300.0) / exact - 1)
        assert closure(s) < 0.01
    report("1_constant_flux", times_s=[1e-8, 1e-7, 1e-6, 1e-5], rel_err=errs)
    assert max(abs(e) for e in errs) < 0.01


# ---------------------------------------------------------------------------------------
# 2. Instantaneous exponential (Beer-law) volumetric source, adiabatic surface
# ---------------------------------------------------------------------------------------
def beer_exact(x, t, F, mu):
    """T - T0 for an instantaneous source F mu exp(-mu x) [J/m^3] in a semi-infinite
    medium with an insulated surface (even extension + Gaussian kernel)."""
    s = np.sqrt(ALPHA * t)
    a = mu * s - x / (2 * s)
    b = mu * s + x / (2 * s)
    g = np.exp(-x ** 2 / (4 * s ** 2))
    term1 = np.where(a > 0, erfcx(a) * g, np.exp(mu ** 2 * s ** 2 - mu * x) * erfc(a))
    return F * mu / (2 * RHO * CP) * (term1 + erfcx(b) * g)


def test_2_exponential_source():
    mat = constant_material("test", RHO, K, CP)
    F, depth = 100.0, 0.5e-6  # J/m^2, 1/mu
    mu = 1 / depth
    tau = 1e-13  # "instantaneous": 1e-13 s pulse
    t_obs = [1e-10, 1e-9, 1e-8, 1e-7]
    s = ThermalSolver(Mesh.build([Layer(mat, 200e-6, 2e-9, 1.05)], 300.0), 300.0,
                      sources=[BeerLawSource(F, SquarePulse(0.0, tau), [mu / RHO])])
    errs_s, errs_prof = [], []
    for t in t_obs:
        s.run(t, dt0=1e-15, dT_target=0.5)
        x = s.x_centers()
        ex = beer_exact(x, t - tau / 2, F, mu)
        errs_s.append((s.surface_temperature() - 300) / beer_exact(0.0, t - tau / 2, F, mu) - 1)
        errs_prof.append(np.max(np.abs(s.T - 300 - ex)) / ex.max())
    report("2_beer_source", times_s=t_obs, surface_rel_err=errs_s, profile_max_err_over_peak=errs_prof,
           closure=closure(s))
    assert max(map(abs, errs_s)) < 0.01 and max(errs_prof) < 0.01 and closure(s) < 0.01


# ---------------------------------------------------------------------------------------
# 3. Two-phase Stefan (Neumann) melting problem
# ---------------------------------------------------------------------------------------
def test_3_stefan_neumann():
    rho, ks, kl, cs, cl, Lf = 1000.0, 2.0, 1.0, 1000.0, 1500.0, 1e5
    Tm, Ti, Tw = 1000.0, 900.0, 1200.0
    als, all_ = ks / (rho * cs), kl / (rho * cl)
    nu = np.sqrt(all_ / als)
    StL, StS = cl * (Tw - Tm) / Lf, cs * (Tm - Ti) / Lf
    f = lambda lam: StL * np.exp(-lam ** 2) / erf(lam) - StS / nu * np.exp(-(nu * lam) ** 2) / erfc(nu * lam) \
        - lam * np.sqrt(np.pi)
    lam = brentq(f, 1e-6, 5.0)
    mat = constant_material("stefan", rho, ks, cs, Tm=Tm, L_f=Lf, dTm=0.5, k_liq=kl, cp_liq=cl)
    s = ThermalSolver(Mesh.build([Layer(mat, 0.02, 2e-6, 1.02, dx_max=10e-6)], Ti), Ti,
                      front=FrontBC("dirichlet", T_wall=Tw))
    errs, t_obs = [], [0.1, 0.5, 1.0]
    for t in t_obs:
        s.run(t, dt0=1e-9, dT_target=2.0)
        X_num = float(np.sum(s.liquid_fraction() * s.dx()))
        X_ex = 2 * lam * np.sqrt(all_ * t)
        errs.append(X_num / X_ex - 1)
    report("3_stefan", lam=lam, times_s=t_obs, front_rel_err=errs, closure=closure(s))
    assert max(map(abs, errs)) < 0.02 and closure(s) < 0.01


# ---------------------------------------------------------------------------------------
# Synthetic ablating case used by tests 4 and 5 (Clausius-Clapeyron vapour, TEST ONLY)
# ---------------------------------------------------------------------------------------
M_W = 0.18384
LV_TEST = 4.5e6  # J/kg, synthetic
TB_TEST = 5800.0  # K, synthetic 1-atm "boiling point"
VAP_TEST = Vapor(p_sat=lambda T: 101325.0 * np.exp(-LV_TEST * M_W / 8.314462618 * (1 / np.asarray(T) - 1 / TB_TEST)),
                 L_v=lambda T: np.full_like(np.asarray(T, float), LV_TEST), molar_mass=M_W,
                 source="synthetic test law")


def ablating_case(dx0, dT_target, dt_fraction, F=20e3, sub=True, dt_rel=0.02):
    armor = constant_material("armor", RHO, K, CP, Tm=3695.0, L_f=2.2e5, dTm=1.0, vapor=VAP_TEST)
    layers = [Layer(armor, 50e-6, dx0, 1.08)]
    if sub:
        steel = constant_material("sub", 7800.0, 30.0, 500.0)
        layers.append(Layer(steel, 1e-3, 1e-6, 1.1))
    s = ThermalSolver(Mesh.build(layers, 600.0), 600.0, back=BackBC("convective", h=1e4, T_coolant=600.0),
                      sources=[BeerLawSource(F, SquarePulse(0.0, 10e-9), [1 / (RHO * 0.1e-6), 1 / (7800 * 0.1e-6)],
                                             dt_fraction=dt_fraction)], alpha_evap=1.0)
    peak = {"T": 0.0}
    s.run(2e-6, dt0=1e-13, dT_target=dT_target, dt_rel=dt_rel,
          record=lambda sv: peak.__setitem__("T", max(peak["T"], sv.surface_temperature())))
    return s, peak["T"]


# ---------------------------------------------------------------------------------------
# 4. Energy conservation with melting, evaporation, rezoning, two layers, coolant
# ---------------------------------------------------------------------------------------
def test_4_energy_conservation():
    s, Tpk = ablating_case(dx0=1e-9, dT_target=5.0, dt_fraction=0.1)
    s.run(1e-2, dt0=1e-9, dT_target=5.0)  # let heat reach the coolant
    L = s.ledger
    res, rel = L.residual()
    report("4_energy_conservation", deposited=L.deposited, vaporized=L.vaporized, to_coolant=L.to_coolant,
           stored_change=L.stored_final - L.stored_initial, residual=res, rel_residual=rel,
           evaporated_kg_m2=s.mass_evaporated, rezones=s.n_rezones, peak_Ts=Tpk)
    assert s.mass_evaporated > 0 and s.n_rezones > 0, "case must exercise evaporation and rezoning"
    assert abs(rel) < 0.01


# ---------------------------------------------------------------------------------------
# 5. Mesh and time-step convergence (halve dx and dt)
# ---------------------------------------------------------------------------------------
def test_5_convergence():
    base, Tb = ablating_case(dx0=2e-9, dT_target=10.0, dt_fraction=0.1, dt_rel=0.02)
    fine, Tf = ablating_case(dx0=1e-9, dT_target=5.0, dt_fraction=0.05, dt_rel=0.01)
    d_b = base.mass_evaporated / RHO
    d_f = fine.mass_evaporated / RHO
    ch_abl = d_b / d_f - 1
    ch_T = Tb / Tf - 1
    report("5_convergence", ablated_nm_base=d_b * 1e9, ablated_nm_fine=d_f * 1e9, ablated_rel_change=ch_abl,
           peakT_base=Tb, peakT_fine=Tf, peakT_rel_change=ch_T)
    assert d_f > 0
    assert abs(ch_abl) < 0.05 and abs(ch_T) < 0.05


# ---------------------------------------------------------------------------------------
# 6. Deposition check: absorbed = incident - transmitted
# ---------------------------------------------------------------------------------------
def test_6_deposition():
    mat = constant_material("test", RHO, K, CP)
    sub = constant_material("sub", 7800.0, 30.0, 500.0)
    # (a) photons through a thin two-layer domain: part of the beam passes through
    mu_rho = [1 / (RHO * 1e-6), 1 / (7800 * 3e-6)]
    s = ThermalSolver(Mesh.build([Layer(mat, 1e-6, 1e-9, 1.1), Layer(sub, 2e-6, 1e-9, 1.1)], 300.0), 300.0,
                      sources=[BeerLawSource(1e3, SquarePulse(0.0, 1e-8), mu_rho)])
    s.run(2e-8, dt0=1e-12)
    tau_tot = 1.0 + 2e-6 / 3e-6
    exp_pass = np.exp(-tau_tot)
    L = s.ledger
    err_a = (L.deposited - (L.incident - L.passed_through)) / L.incident
    err_pass = L.passed_through / L.incident / exp_pass - 1
    # (b) 1 MeV alphas in W: CSDA end point vs NIST range; thin domain transmits the rest
    astar = StoppingTable.nist("ASTAR", "074")
    E0 = 1.0 * MEV
    R_nist = float(astar.range(E0)) / 19300.0  # m
    thick = ThermalSolver(Mesh.build([Layer(mat, 3 * R_nist, 1e-9, 1.05)], 300.0), 300.0)
    src = MonoIonSource(1e3, SquarePulse(0, 1e-8), E0, [astar])
    fr = src.fractions(thick)
    bounds = np.r_[0.0, np.cumsum(thick.m)] / 19300.0
    x_end = bounds[np.nonzero(fr)[0][-1] + 1]
    dx_end = np.diff(bounds)[np.nonzero(fr)[0][-1]]
    thin = ThermalSolver(Mesh.build([Layer(mat, 0.5 * R_nist, 1e-9, 1.05)], 300.0), 300.0)
    fr_thin = src.fractions(thin)
    E_out = float(astar.energy_at_range(astar.range(E0) - thin.m.sum()))
    first = fr[0] * E0 / (float(np.interp(E0, astar.E, astar.S)) * thick.m[0])
    report("6_deposition", beer_balance_err=err_a, beer_passthrough_rel_err=err_pass,
           alpha_sum_fractions_thick=fr.sum(), alpha_end_um=x_end * 1e6, nist_csda_range_um=R_nist * 1e6,
           alpha_thin_balance_err=fr_thin.sum() + E_out / E0 - 1, first_cell_vs_S_ratio=first)
    assert abs(err_a) < 1e-12 and abs(err_pass) < 1e-6
    assert abs(fr.sum() - 1) < 1e-12
    assert abs(x_end - R_nist) <= dx_end
    assert abs(fr_thin.sum() + E_out / E0 - 1) < 1e-12
    assert abs(first - 1) < 0.01
