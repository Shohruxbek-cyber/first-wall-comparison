"""Analytical verification tests (phase 4 of the pre-submission verification suite, 17 Oct 2026).

These EXTEND tests/test_verification.py (whose six tests stay the gate for production runs) and
tests/test_alloys.py test 6; nothing here duplicates an existing assertion:

  test_1_*  extends verification test 1 (constant surface flux: surface value at four times, 1 %) with the
            full erfc profile, a uniform-dt and a graded-dt (dt_rel) time-refinement study and a nested
            geometric-mesh refinement study, each with the observed order;
  test_2_*  NEW: instantaneous surface energy deposition (plane-source / Gaussian limit), implemented as a thin
            UniformDepthSource slab released in a 1e-13 s square pulse; surface and profile at four times, the
            finite thickness and duration accounted for exactly, time and mesh refinement with observed orders;
  test_3_*  extends verification test 3 (TWO-phase Neumann, dTm = 0.5 K, front position 2 %) with the ONE-phase
            Neumann problem (solid at the melting point, St = 1) at the DEFAULT dTm = 1 K: front position,
            liquid profile, cumulative surface heat, closure, time and mesh refinement, and the effect of the
            mushy interval quantified against the sharp-interface solution;
  test_4_*  NEW: pure conduction, no evaporation, closed system: energy stored is conserved / equals the
            deposited energy at rounding level, final temperature equals the exact mixing temperature
            (verification test 4 and damage test 4 check closure to 1e-2 / 1e-6 with evaporation, rezoning
            and production sources);
  test_5_*  extends alloys test 6 (production W case, sources removed, 20 us, 1e-9 K) to 1e4 uniform steps
            over 1 s: constant material and library.tungsten(), adiabatic and convective-at-equilibrium back
            face, evaporation off and on (drift compared with the Hertz-Knudsen-Langmuir prediction).

Nothing here writes to runs/ or results/ (no report() helper).  Run with `pytest -s` to see the numbers.
"""
import numpy as np
import pytest
from scipy.special import erf

from ife_wall.library import tungsten
from ife_wall.materials import constant_material
from ife_wall.source_term import SquarePulse, UniformDepthSource, VolumetricUniformSource
from ife_wall.thermal import BackBC, FrontBC, Layer, Mesh, ThermalSolver

from _analytical_solutions import (constant_flux_profile, constant_flux_surface, nested_geometric_mesh,
                                   neumann_front, neumann_heat_input, neumann_liquid_profile,
                                   neumann_one_phase_lambda, neumann_two_phase_lambda, observed_orders,
                                   plane_source_profile, slab_source_profile, square_pulse_surface_exact)

RHO, K, CP = 19300.0, 100.0, 140.0  # test-only W-like magnitudes (the same as tests/test_verification.py)
ALPHA = K / (RHO * CP)
T0 = 300.0
BIG = 1e9  # dT_target / dt_rel value that switches the corresponding step control off (linear problems)


def say(title, **vals):
    def fmt(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, float):
            return f"{v:.4e}" if (abs(v) < 1e-2 or abs(v) >= 1e4) and v != 0 else f"{v:.6g}"
        return str(v)
    print(f"\n[{title}] " + ", ".join(f"{k}={fmt(v)}" for k, v in vals.items()))


# =============================================================================================
# 1. Semi-infinite solid, constant surface flux  (Carslaw & Jaeger 2.9; Incropera & DeWitt Ch. 5 case 2)
# =============================================================================================
Q_FLUX = 1e9  # W/m^2
T_END_1 = 1e-6  # s
L_1 = 20 * np.sqrt(ALPHA * T_END_1)  # 20 diffusion lengths: the back face is undisturbed
MESH_DX0_1, MESH_R0_1 = 50e-9, 1.21  # level-0 member of the nested family (33 cells); level l has 2^l x more


def _flux_solver(level):
    dx0, r = nested_geometric_mesh(level, MESH_DX0_1, MESH_R0_1)
    mat = constant_material("test", RHO, K, CP)
    return ThermalSolver(Mesh.build([Layer(mat, L_1, dx0, r)], T0), T0,
                         front=FrontBC("flux", flux=lambda t: Q_FLUX))


def _flux_errors(s):
    """Relative errors at t_end: surface, profile max-norm and L2 (relative to the exact surface rise)."""
    t = s.t
    x = s.x_centers()
    ex = constant_flux_profile(x, t, Q_FLUX, K, ALPHA)
    Ts = constant_flux_surface(t, Q_FLUX, K, ALPHA)
    prof = s.T - T0 - ex
    dx = s.dx()
    return dict(rel_s=(s.surface_temperature() - T0 - Ts) / Ts, linf=float(np.abs(prof).max() / Ts),
                l2=float(np.sqrt(np.sum(prof ** 2 * dx) / np.sum(ex ** 2 * dx))),
                closure=s.ledger.residual()[1], n=len(s.m), steps=s.n_steps_accepted)


def test_1a_constant_flux_profile():
    """Full profile (erfc) at t_end on the level-3 mesh (262 cells, first cell 5.7 nm) with 800 uniform steps.
    Expected error of the scheme: backward Euler first order, measured coefficient 0.12/N (test_1b) -> 1.5e-4,
    plus the spatial error of level 3, 4e-5 (test_1d): about 2e-4.  Tolerance: twice that, 4e-4."""
    s = _flux_solver(3)
    dt = T_END_1 / 800
    s.run(T_END_1, dt0=dt, dt_max=dt, dT_target=BIG, dt_rel=BIG)
    e = _flux_errors(s)
    say("1a_profile", **e)
    assert abs(e["rel_s"]) < 4e-4 and e["linf"] < 4e-4 and e["l2"] < 6e-4
    assert abs(e["closure"]) < 1e-12


def test_1b_constant_flux_time_order_uniform_dt():
    """Uniform time steps N = 50..800 on the level-4 mesh (524 cells): the error must fall as 1/N (backward
    Euler).  The spatial error of level 4 is ~1e-5 (test_1d), below the finest time error (1.5e-4), so the
    observed order between successive N must be 1 within [0.9, 1.15]."""
    Ns = [50, 100, 200, 400, 800]
    errs = []
    for N in Ns:
        s = _flux_solver(4)
        dt = T_END_1 / N
        s.run(T_END_1, dt0=dt, dt_max=dt, dT_target=BIG, dt_rel=BIG)
        e = _flux_errors(s)
        assert s.n_steps_accepted == N
        errs.append(e)
    rel_s = [e["rel_s"] for e in errs]
    linf = [e["linf"] for e in errs]
    orders_s, orders_p = observed_orders(rel_s), observed_orders(linf)
    say("1b_uniform_dt", N=Ns, rel_s=rel_s, linf=linf, order_surface=orders_s, order_profile=orders_p,
        coefficient_relerr_times_N=[r * N for r, N in zip(rel_s, Ns)])
    assert all(0.9 <= o <= 1.15 for o in orders_s), orders_s
    assert all(0.9 <= o <= 1.15 for o in orders_p), orders_p


def test_1c_constant_flux_time_order_graded_dt_rel():
    """The solver's own control dt <= dt_rel * (t - t_event) (steps proportional to the elapsed time, the
    natural choice for the sqrt(t) transient): halving dt_rel must halve the error, observed order in [0.85, 1.15]."""
    drs = [0.08, 0.04, 0.02, 0.01]
    errs = []
    for dr in drs:
        s = _flux_solver(4)
        s.run(T_END_1, dt0=1e-13, dT_target=BIG, dt_rel=dr)
        errs.append(_flux_errors(s))
    rel_s = [e["rel_s"] for e in errs]
    linf = [e["linf"] for e in errs]
    orders_s, orders_p = observed_orders(rel_s), observed_orders(linf)
    say("1c_dt_rel", dt_rel=drs, steps=[e["steps"] for e in errs], rel_s=rel_s, linf=linf,
        order_surface=orders_s, order_profile=orders_p, coefficient_relerr_over_dt_rel=[r / d for r, d in zip(rel_s, drs)])
    assert all(0.85 <= o <= 1.15 for o in orders_s), orders_s
    assert all(0.85 <= o <= 1.15 for o in orders_p), orders_p


def test_1d_constant_flux_mesh_order():
    """Nested geometric meshes, levels 0-4 (33 -> 524 cells; r = 1.21 -> 1.012; first cell 50 -> 2.9 nm).
    The time error is removed by first-order Richardson extrapolation between N = 400 and N = 800 uniform
    steps (T_ext = 2 T_800 - T_400, residual O(dt^2) ~ 1e-6), leaving the spatial error, which must fall 4x per
    level (second order).  Two measures: (i) the extrapolated surface error against the exact solution,
    (ii) the differences between successive levels (independent of any time-error floor).
    Expected: order 2 on this smoothly graded family (Manteuffel & White 1986 supra-convergence of the 3-point
    scheme on smooth non-uniform meshes); asserted within [1.7, 2.3] for (i) over the levels where the
    extrapolated error is above the O(dt^2) residual, and [1.8, 2.2] for (ii)."""
    levels = [0, 1, 2, 3, 4]
    Ts_ext, errs_ext, ncells = [], [], []
    for lev in levels:
        vals = {}
        for N in (400, 800):
            s = _flux_solver(lev)
            dt = T_END_1 / N
            s.run(T_END_1, dt0=dt, dt_max=dt, dT_target=BIG, dt_rel=BIG)
            vals[N] = s.surface_temperature() - T0
            n = len(s.m)
        T_ext = 2 * vals[800] - vals[400]
        Ts_ext.append(T_ext)
        errs_ext.append(T_ext / constant_flux_surface(T_END_1, Q_FLUX, K, ALPHA) - 1)
        ncells.append(n)
    diffs = [abs(Ts_ext[i] - Ts_ext[i + 1]) for i in range(len(levels) - 1)]
    orders_ext = observed_orders(errs_ext)
    orders_diff = observed_orders(diffs)
    say("1d_mesh", level=levels, cells=ncells, extrapolated_rel_err=errs_ext, order_vs_exact=orders_ext,
        successive_diff_K=diffs, order_successive=orders_diff)
    # (i) levels 0-3: extrapolated errors 2e-3 .. 4e-5 are well above the ~1e-6 Richardson residual
    assert all(1.7 <= o <= 2.3 for o in orders_ext[:3]), orders_ext
    # (ii) all successive differences
    assert all(1.8 <= o <= 2.2 for o in orders_diff), orders_diff
    assert abs(errs_ext[-1]) < 1e-4


# =============================================================================================
# 2. Instantaneous surface energy deposition (plane source at an insulated surface; C&J 10.3 + image)
# =============================================================================================
Q_DEP, D_DEP, TAU_DEP = 2.0, 4e-9, 1e-13  # J/m^2, slab thickness, pulse length (dt_limit = 0.1 tau in the pulse)
T_OBS_2 = [1e-10, 1e-9, 1e-8, 1e-7]
L_2 = 60e-6  # > 20 sqrt(alpha t) at 1e-7 s
MESH_DX0_2, MESH_R0_2, MESH_DXMAX0_2 = 2e-9, 1.21, 1e-6


def _dep_solver(level):
    r, dx0, dxm = MESH_R0_2, MESH_DX0_2, MESH_DXMAX0_2
    dx0, r = nested_geometric_mesh(level, dx0, r)
    dxm = dxm / 2 ** level
    mat = constant_material("test", RHO, K, CP)
    src = UniformDepthSource(Q_DEP, SquarePulse(0.0, TAU_DEP), depth=D_DEP, rho0=RHO)
    s = ThermalSolver(Mesh.build([Layer(mat, L_2, dx0, r, dx_max=dxm)], T0), T0, sources=[src])
    s.run(TAU_DEP, dt0=1e-15, dT_target=BIG, dt_rel=BIG)  # the deposition: 10 steps of 0.1 tau
    return s


def _dep_errors(s, t):
    """Errors at the observation time t against (i) the plane-source (delta) solution and (ii) the exact
    finite-slab solution; both evaluated at t - tau/2 (the residual of that shift is O((tau/t)^2) <= 1e-6)."""
    te = t - TAU_DEP / 2
    x = s.x_centers()
    rc = RHO * CP
    Ts_num = s.surface_temperature() - T0
    Ts_d, Ts_s = plane_source_profile(0.0, te, Q_DEP, rc, ALPHA), slab_source_profile(0.0, te, Q_DEP, D_DEP, rc, ALPHA)
    ex_s = slab_source_profile(x, te, Q_DEP, D_DEP, rc, ALPHA)
    ex_d = plane_source_profile(x, te, Q_DEP, rc, ALPHA)
    prof = s.T - T0
    dx = s.dx()
    return dict(t=t, Ts_num=Ts_num, Ts_delta=Ts_d, Ts_slab=Ts_s, rel_s_delta=Ts_num / Ts_d - 1, rel_s_slab=Ts_num / Ts_s - 1,
                linf_delta=float(np.abs(prof - ex_d).max() / Ts_d), linf_slab=float(np.abs(prof - ex_s).max() / Ts_s),
                l2_slab=float(np.sqrt(np.sum((prof - ex_s) ** 2 * dx) / np.sum(ex_s ** 2 * dx))),
                thickness_bias=Ts_s / Ts_d - 1, closure=s.ledger.residual()[1], steps=s.n_steps_accepted, n=len(s.m))


def test_2a_plane_source_surface_and_profile():
    """Level-2 mesh (351 cells: first cell 0.47 nm, r = 1.049, dx_max 0.25 um), dt_rel = 0.01 after the pulse.
    Checks at 1e-10..1e-7 s: surface rise and profile against the delta solution (with the known finite-thickness
    bias -d^2/(12 alpha t), 3.6e-4 at 1e-10 s) and against the exact slab solution (bias-free).
    Expected error of the scheme against the slab solution: graded first-order time error, measured coefficient
    0.16-0.19 x dt_rel (test_2b) -> 1.9e-3, plus the spatial error of level 2 (1.4e-4, test_2c): about 2e-3.
    Tolerance: twice that, 4e-3."""
    s = _dep_solver(2)
    rows = []
    for t in T_OBS_2:
        s.run(t, dt0=1e-15, dT_target=BIG, dt_rel=0.01)
        rows.append(_dep_errors(s, t))
    say("2a_surface_profile", n=rows[0]["n"], steps=rows[-1]["steps"], t=[r["t"] for r in rows],
        Ts_num_K=[r["Ts_num"] for r in rows], Ts_delta_K=[r["Ts_delta"] for r in rows],
        rel_s_delta=[r["rel_s_delta"] for r in rows], rel_s_slab=[r["rel_s_slab"] for r in rows],
        thickness_bias=[r["thickness_bias"] for r in rows], linf_delta=[r["linf_delta"] for r in rows],
        linf_slab=[r["linf_slab"] for r in rows], l2_slab=[r["l2_slab"] for r in rows], closure=rows[-1]["closure"],
        deposited_J_m2=s.ledger.deposited, duration_shift_residual=[
            square_pulse_surface_exact(t, TAU_DEP, Q_DEP, RHO * CP, ALPHA) / plane_source_profile(0.0, t - TAU_DEP / 2, Q_DEP, RHO * CP, ALPHA) - 1
            for t in T_OBS_2])
    assert abs(s.ledger.deposited - Q_DEP) < 1e-12 * Q_DEP
    for r in rows:
        assert abs(r["rel_s_slab"]) < 4e-3 and r["linf_slab"] < 4e-3, r
        # against the delta solution the finite-thickness bias adds; it is known exactly
        assert abs(r["rel_s_delta"] - r["thickness_bias"]) < 4e-3, r
    assert abs(rows[-1]["closure"]) < 1e-12


def test_2b_plane_source_time_order():
    """dt_rel = 0.08..0.01 on the level-3 mesh (703 cells): surface and profile errors (against the slab
    solution) at each observation time must fall as dt_rel (first order); the spatial error of level 3
    (4e-5, test_2c) is far below the finest time error (1.6e-3), so the observed order must be 1 within
    [0.85, 1.15] at every observation time."""
    drs = [0.08, 0.04, 0.02, 0.01]
    table = {t: [] for t in T_OBS_2}
    steps = []
    for dr in drs:
        s = _dep_solver(3)
        for t in T_OBS_2:
            s.run(t, dt0=1e-15, dT_target=BIG, dt_rel=dr)
            table[t].append(_dep_errors(s, t))
        steps.append(s.n_steps_accepted)
    out = {}
    for t in T_OBS_2:
        rs = [r["rel_s_slab"] for r in table[t]]
        li = [r["linf_slab"] for r in table[t]]
        out[t] = (rs, li, observed_orders(rs), observed_orders(li))
        say(f"2b_dt_rel_t={t:.0e}", dt_rel=drs, steps=steps, rel_s_slab=rs, linf_slab=li, order_surface=out[t][2],
            order_profile=out[t][3])
    for t in T_OBS_2:
        assert all(0.85 <= o <= 1.15 for o in out[t][2]), (t, out[t][2])
        assert all(0.85 <= o <= 1.15 for o in out[t][3]), (t, out[t][3])


def test_2c_plane_source_mesh_order():
    """Nested meshes, levels 0-4 (88 -> 1406 cells; first cell 2 -> 0.11 nm, dx_max 1 um -> 62 nm), fixed
    dt_rel = 0.02.  Successive-level differences of the surface rise at each observation time give the spatial
    order (the common first-order time error, 3e-3, cancels); expected 2, asserted in [1.8, 2.2] at every
    observation time.  The spatial error of level l is estimated as 4/3 of the difference to level l+1 (sum of
    a 1/4 geometric series), reported relative to the exact surface rise.  The error against the exact slab
    solution at level 4 must be below 2x the expected time error (2 x 0.19 x 0.02 = 7.6e-3)."""
    levels = [0, 1, 2, 3, 4]
    table = {t: [] for t in T_OBS_2}
    cells = []
    for lev in levels:
        s = _dep_solver(lev)
        for t in T_OBS_2:
            s.run(t, dt0=1e-15, dT_target=BIG, dt_rel=0.02)
            table[t].append(_dep_errors(s, t))
        cells.append(len(s.m))
    for t in T_OBS_2:
        Ts = [r["Ts_num"] for r in table[t]]
        diffs = [abs(Ts[i] - Ts[i + 1]) for i in range(len(Ts) - 1)]
        od = observed_orders(diffs)
        rel = [r["rel_s_slab"] for r in table[t]]
        li = [r["linf_slab"] for r in table[t]]
        say(f"2c_mesh_t={t:.0e}", level=levels, cells=cells, rel_s_slab=rel, linf_slab=li,
            successive_diff_K=diffs, order_successive=od,
            spatial_rel_err_estimate=[4.0 / 3.0 * d / table[t][0]["Ts_slab"] for d in diffs])
        assert all(1.8 <= o <= 2.2 for o in od), (t, od)
        assert abs(rel[-1]) < 7.6e-3, (t, rel)


# =============================================================================================
# 3. One-phase Stefan (Neumann) melting problem  (Carslaw & Jaeger 11.2; Alexiades & Solomon 2.1)
# =============================================================================================
RHO_S, K_S, C_S, LF_S = 1000.0, 1.0, 1000.0, 1e5  # alpha = 1e-6 m^2/s, test-only
TM_S, TS_S = 1000.0, 1100.0  # melting point and imposed surface temperature: St = c (Ts - Tm) / L = 1
ALPHA_S = K_S / (RHO_S * C_S)
DTM_S = 1.0  # the DEFAULT mushy interval of Material / constant_material
T_OBS_3 = [0.01, 0.1, 0.5, 1.0]
LAM_NOM = neumann_one_phase_lambda(C_S * (TS_S - TM_S) / LF_S)  # sharp interface at Tm
LAM_SOL = neumann_one_phase_lambda(C_S * (TS_S - (TM_S - DTM_S / 2)) / LF_S)  # sharp interface at the solidus
LAM_2PH = neumann_two_phase_lambda(C_S * (TS_S - TM_S) / LF_S, C_S * (DTM_S / 2) / LF_S)  # solid initially at solidus, front at Tm


def _stefan_run(dx, dT_target, dt_rel, L=5e-3):
    """Uniform mesh of width dx; the solid starts at the solidus Tm - dTm/2 (liquid fraction exactly 0)."""
    mat = constant_material("stefan1", RHO_S, K_S, C_S, Tm=TM_S, L_f=LF_S, dTm=DTM_S)
    Ti = TM_S - DTM_S / 2
    s = ThermalSolver(Mesh.build([Layer(mat, L, dx, 1.0)], Ti), Ti, front=FrontBC("dirichlet", T_wall=TS_S))
    rows = []
    for t in T_OBS_3:
        s.run(t, dt0=1e-9, dT_target=dT_target, dt_rel=dt_rel)
        f = s.liquid_fraction()
        x, dxs = s.x_centers(), s.dx()
        X_int = float(np.sum(f * dxs))  # melted thickness from the latent-heat budget
        X_thr = s.melt_depth()  # contiguous cells with f >= 0.5
        liq = f >= 1.0 - 1e-9
        row = dict(t=t, X_int=X_int, X_thr=X_thr, Ti=Ti, T_back=float(s.T[-1]), steps=s.n_steps_accepted,
                   rejected=s.n_steps_rejected, newton_max=s.newton_iters_max_step, closure=s.ledger.residual()[1],
                   Q_num=s.ledger.front_flux_in)
        for tag, lam in (("nom", LAM_NOM), ("sol", LAM_SOL)):
            Tm_eff = TM_S if tag == "nom" else TM_S - DTM_S / 2
            X_ex = neumann_front(t, lam, ALPHA_S)
            row[f"rel_X_{tag}"] = X_int / X_ex - 1
            row[f"rel_Xthr_{tag}"] = X_thr / X_ex - 1
            T_ex = neumann_liquid_profile(x, t, TS_S, Tm_eff, lam, ALPHA_S)
            row[f"prof_{tag}"] = float(np.abs(s.T[liq] - T_ex[liq]).max() / (TS_S - TM_S)) if liq.any() else np.nan
            row[f"rel_Q_{tag}"] = s.ledger.front_flux_in / neumann_heat_input(t, K_S, TS_S, Tm_eff, lam, ALPHA_S) - 1
        rows.append(row)
    return dict(n=len(s.m), dx=dx, dT_target=dT_target, dt_rel=dt_rel, rows=rows)


def _say_stefan(title, d):
    r = d["rows"]
    say(title, n=d["n"], dx=d["dx"], dT_target=d["dT_target"], dt_rel=d["dt_rel"], t=[q["t"] for q in r],
        X_int_mm=[q["X_int"] * 1e3 for q in r], rel_X_vs_Tm=[q["rel_X_nom"] for q in r],
        rel_X_vs_solidus=[q["rel_X_sol"] for q in r], rel_Xthreshold_vs_Tm=[q["rel_Xthr_nom"] for q in r],
        liquid_profile_err_vs_Tm=[q["prof_nom"] for q in r], liquid_profile_err_vs_solidus=[q["prof_sol"] for q in r],
        rel_Q_vs_Tm=[q["rel_Q_nom"] for q in r], rel_Q_vs_solidus=[q["rel_Q_sol"] for q in r],
        steps=[q["steps"] for q in r], rejected=r[-1]["rejected"], newton_max=r[-1]["newton_max"],
        closure=[q["closure"] for q in r], T_back_minus_Ti=[q["T_back"] - q["Ti"] for q in r])


def test_3a_stefan_one_phase_front_profile_heat():
    """Baseline resolution dx = 10 um (500 cells over 5 mm), dT_target = 2 K, dt_rel = 0.02 (the existing
    verification test 3 uses 2-10 um, 2 K, 0.02).  Compared with the sharp-interface Neumann solution at the
    nominal Tm: the mushy interval [Tm - 0.5, Tm + 0.5] K puts the converged enthalpy-method front within the
    band spanned by the sharp problems 'melting point at the solidus' (+2.0e-3 for dTm = 1 K) and 'solid
    initially at the solidus, front at Tm' (-3.2e-3); both are computed and printed.  The discretisation error
    at this resolution is <= 1e-3 (test_3b/3c).  Tolerance for the front position (latent-heat budget) and the
    surface heat: 5e-3 = band + 2x the expected discretisation error.  Liquid profile (max over the fully
    liquid cells, relative to Ts - Tm = 100 K): the liquid cell next to the mushy zone sits at the liquidus,
    up to dTm/2 = 0.5 K off the sharp profile, the front position within a cell adds |dT/dx| dx/2 = 1 K at
    0.1 s (2e5 K/m), the time error ~0.3 K: expected up to ~1.8e-2, tolerance 2e-2 (the error against the
    solidus-sharp profile is printed as well)."""
    d = _stefan_run(10e-6, 2.0, 0.02)
    _say_stefan("3a_baseline", d)
    say("3a_lambda", St=C_S * (TS_S - TM_S) / LF_S, lam_nominal=LAM_NOM, lam_solidus=LAM_SOL,
        band_solidus=LAM_SOL / LAM_NOM - 1, band_two_phase=LAM_2PH / LAM_NOM - 1)
    for q in d["rows"]:
        assert abs(q["T_back"] - q["Ti"]) < 1e-9  # the solid stays at the solidus: a one-phase problem
        assert abs(q["closure"]) < 1e-10
        if q["t"] >= 0.1:
            assert abs(q["rel_X_nom"]) < 5e-3, q
            assert abs(q["rel_Q_nom"]) < 5e-3, q
            assert q["prof_nom"] < 2e-2, q


def test_3b_stefan_one_phase_time_refinement():
    """dx = 10 um fixed; (dT_target, dt_rel) halved four times.  (i) The error against the solidus-sharp
    solution at t >= 0.1 s must be < 1.5e-3 at the finest level (expected: ~0.02*dt_rel + 0.01*dx/s(t) <= 7e-4).
    (ii) Self-convergence: successive differences of the front position must fall by ~2x (first order);
    asserted order in [0.6, 1.5] for the last two pairs at t = 0.5 and 1 s (the front crossing cell boundaries
    adds a non-smooth O(dx) component that limits the smoothness of the observed order)."""
    ctrl = [(8.0, 0.08), (4.0, 0.04), (2.0, 0.02), (1.0, 0.01)]
    runs = [_stefan_run(10e-6, a, b) for a, b in ctrl]
    for d in runs:
        _say_stefan(f"3b_time_dT{d['dT_target']}_dtrel{d['dt_rel']}", d)
    for i, t in enumerate(T_OBS_3):
        X = [d["rows"][i]["X_int"] for d in runs]
        diffs = [abs(X[j] - X[j + 1]) for j in range(len(X) - 1)]
        od = observed_orders(diffs)
        e_sol = [d["rows"][i]["rel_X_sol"] for d in runs]
        say(f"3b_order_t={t}", rel_X_vs_solidus=e_sol, successive_diff_um=[v * 1e6 for v in diffs], order_successive=od)
        if t >= 0.5:
            assert all(0.6 <= o <= 1.5 for o in od[-2:]), (t, od)
        if t >= 0.1:
            assert abs(e_sol[-1]) < 1.5e-3, (t, e_sol)


def test_3c_stefan_one_phase_mesh_refinement():
    """dx = 40, 20, 10 um at fixed (dT_target, dt_rel) = (1 K, 0.01).  The mushy zone is w ~ 2 dTm / |dT/dx|_front
    = 6-30 um wide (0.01-1 s), i.e. under-resolved on the coarse meshes, so monotone convergence is not expected;
    reported: error against the solidus-sharp solution per mesh and the successive differences.
    Asserted: |error| < 2e-3 at t >= 0.1 s on every mesh (expected ~0.01*dx/s(t) <= 1e-3 at 40 um, 0.1 s)."""
    runs = [_stefan_run(dx, 1.0, 0.01) for dx in (40e-6, 20e-6, 10e-6)]
    for d in runs:
        _say_stefan(f"3c_mesh_dx{d['dx']*1e6:.0f}um", d)
    for i, t in enumerate(T_OBS_3):
        X = [d["rows"][i]["X_int"] for d in runs]
        diffs = [abs(X[j] - X[j + 1]) for j in range(len(X) - 1)]
        e_sol = [d["rows"][i]["rel_X_sol"] for d in runs]
        say(f"3c_order_t={t}", rel_X_vs_solidus=e_sol, successive_diff_um=[v * 1e6 for v in diffs],
            order_successive=observed_orders(diffs))
        if t >= 0.1:
            assert all(abs(e) < 2e-3 for e in e_sol), (t, e_sol)


# =============================================================================================
# 4. Energy conservation, pure conduction, no evaporation (closed system; rounding-level closure)
# =============================================================================================
EPS = np.finfo(float).eps


def _mixing_temperature(s, E_total):
    """Uniform temperature with the same total enthalpy sum_i m_i h_i(T) (exact for any h(T) table)."""
    from scipy.optimize import brentq
    mats = s.materials
    f = lambda T: sum(float(np.sum(s.m[s.layer == j]) * mats[j].h_of_T(np.array([T]))[0]) for j in range(len(mats))) - E_total
    return brentq(f, 250.0, 5000.0, xtol=1e-12)


def _two_layer(melt):
    w = constant_material("w", RHO, K, CP, Tm=1200.0, L_f=2e5, dTm=1.0) if melt else constant_material("w", RHO, K, CP)
    st = constant_material("steel", 7800.0, 30.0, 500.0)
    mesh = Mesh.build([Layer(w, 100e-6, 1e-7, 1.1, dx_max=5e-6), Layer(st, 1e-3, 1e-6, 1.1, dx_max=20e-6)], 300.0)
    T_init = np.where(mesh.layer == 0, 1500.0, 300.0)  # step: hot (liquid, if melt) W layer on cold steel
    return ThermalSolver(mesh, T_init)  # front and back adiabatic, no vapour


@pytest.mark.parametrize("melt", [False, True])
def test_4a_closed_two_layer_slab_energy_and_mixing_temperature(melt):
    """Two-layer slab (100 um W-like on 1 mm steel-like), both faces adiabatic, step initial temperature
    1500/300 K, run 1 s (>> L^2/alpha = 0.13 s): the stored energy must not change and the final temperature
    must be uniform at the exact mixing temperature.  'Machine precision' for this solver: the Newton iterate
    satisfies the cell balances to rounding (piecewise-linear h(T): the last iteration solves the linear system
    exactly), and the internal fluxes telescope exactly, so |dE| ~ eps sqrt(N_steps) sum|m h| ~ 5e-9 J/m^2 on
    3e5 J/m^2 (1e-14 relative); the Newton tolerance tol_T = 1e-6 K would correspond to 5e-4 J/m^2 (1.6e-9)
    if the residual were not driven to rounding.  Tolerance: 1e-12 of the energy exchanged between the layers
    (100x the rounding estimate, 1000x below the Newton-tolerance level)."""
    s = _two_layer(melt)
    E0 = s.stored_energy()
    s.run(1.0, dt0=1e-10, dT_target=5.0, dt_rel=0.02)
    E1 = s.stored_energy()
    exchanged = s.ledger.through_interface[0]
    T_mix = _mixing_temperature(s, E0)
    dev = float(np.abs(s.T - T_mix).max())
    res_abs, res_rel_ledger = s.ledger.residual()
    say(f"4a_two_layer_melt={melt}", n=len(s.m), steps=s.n_steps_accepted, rejected=s.n_steps_rejected,
        newton_max=s.newton_iters_max_step, E0_J_m2=E0, dE_J_m2=E1 - E0, exchanged_J_m2=exchanged,
        dE_over_exchanged=(E1 - E0) / exchanged, dE_over_E0=(E1 - E0) / abs(E0),
        rounding_estimate_J_m2=EPS * np.sqrt(s.n_steps_accepted) * float(np.sum(np.abs(s.m * s.H))),
        ledger_residual_J_m2=res_abs, ledger_relative_residual=res_rel_ledger, T_mix=T_mix, max_dev_from_T_mix_K=dev,
        final_liquid_fraction_sum=float(s.liquid_fraction().sum()))
    assert exchanged > 1e5
    assert abs(E1 - E0) < 1e-12 * exchanged
    assert dev < 1e-9
    assert float(s.liquid_fraction().sum()) == 0.0  # 470 K < Tm: everything re-solidified


@pytest.mark.parametrize("kind", ["depth", "volumetric"])
def test_4b_insulated_slab_volumetric_source(kind):
    """200 um slab, both faces adiabatic, 1e4 J/m^2 deposited in 10 ns uniformly over the first 5 um
    (UniformDepthSource) or over the whole slab (VolumetricUniformSource); run to 1e-2 s (10 L^2/alpha).
    The stored energy change must equal the deposited energy (ledger) to rounding, and the final temperature
    the exact T0 + F / sum(m cp).  Tolerance 1e-12 relative (as in test_4a) and 1e-9 K."""
    w = constant_material("w", RHO, K, CP)
    mesh = Mesh.build([Layer(w, 200e-6, 2e-8, 1.1, dx_max=2e-6)], T0)
    F = 1e4
    src = UniformDepthSource(F, SquarePulse(0.0, 1e-8), depth=5e-6, rho0=RHO) if kind == "depth" \
        else VolumetricUniformSource(F, SquarePulse(0.0, 1e-8))
    s = ThermalSolver(mesh, T0, sources=[src])
    E0 = s.stored_energy()
    s.run(1e-2, dt0=1e-12, dT_target=5.0, dt_rel=0.02)
    E1 = s.stored_energy()
    dep = s.ledger.deposited
    T_eq = T0 + F / float(np.sum(s.m * CP))
    dev = float(np.abs(s.T - T_eq).max())
    say(f"4b_source_{kind}", n=len(s.m), steps=s.n_steps_accepted, deposited_J_m2=dep, incident_J_m2=s.ledger.incident,
        dE_J_m2=E1 - E0, dE_minus_deposited_over_deposited=(E1 - E0 - dep) / dep, ledger_relative_residual=s.ledger.residual()[1],
        T_eq=T_eq, max_dev_from_T_eq_K=dev)
    assert abs(dep - F) < 1e-12 * F
    assert abs(E1 - E0 - dep) < 1e-12 * dep
    assert dev < 1e-9


# =============================================================================================
# 5. Zero-flux invariance over 1e4 uniform steps (evaporation off and on)
# =============================================================================================
ZERO_CASES = [
    ("constant", 1000.0, "adiabatic", False),
    ("constant", 1000.0, "convective_eq", False),
    ("W", 1500.0, "convective_eq", False),
    ("W", 1500.0, "convective_eq", True),
    ("W", 3000.0, "adiabatic", False),
    ("W", 3000.0, "adiabatic", True),
]


@pytest.mark.parametrize("mat_name,T_init,back_kind,evap", ZERO_CASES, ids=[f"{m}-{int(T)}K-{b}-evap{'On' if e else 'Off'}" for m, T, b, e in ZERO_CASES])
def test_5_zero_flux_invariance(mat_name, T_init, back_kind, evap):
    """Uniform initial temperature, no sources, front adiabatic, back adiabatic or convective at the coolant
    temperature (h = 1e4), 1e4 uniform steps of 1e-4 s (dt0 = dt_max, dT_target and dt_rel switched off).
    Evaporation off: the temperature must not change at all (tolerance 1e-12 K ~ 5x the rounding of T).
    Evaporation on (library.tungsten(), Raffray 2005 vapour pressure): the ONLY change allowed is the
    Hertz-Knudsen-Langmuir cooling.  1500 K: p_sat = 1.2e-17 Pa, j = 1.9e-20 kg/m^2/s, latent loss 7e-14 J/m^2
    over 1 s, below the enthalpy resolution of the slab (predicted dT ~ 3e-16 K) -> still 1e-12 K, and the
    evaporated mass must equal j(T0) t to 1e-9.  3000 K, adiabatic: p_sat = 8.2e-3 Pa, j = 8.8e-6 kg/m^2/s,
    35 W/m^2 latent: the slab (100 um) cools uniformly by E_latent / (m c) ~ 0.086 K; the mean temperature must
    match the enthalpy-table prediction T(h0 - E_latent/m) to 1e-3 of the drift (quasi-uniform: the conduction
    drop across the slab is q L / 2k = 2e-5 K, 2e-4 of the drift), the mass to 2e-3 (j falls 0.1 % as T drops)
    and the ledger must close to 1e-6 of the vaporized energy."""
    if mat_name == "constant":
        mat = constant_material("test", RHO, K, CP)
    else:
        mat = tungsten()
    back = BackBC() if back_kind == "adiabatic" else BackBC("convective", h=1e4, T_coolant=T_init)
    mesh = Mesh.build([Layer(mat, 100e-6, 1e-9, 1.2, dx_max=5e-6)], T_init)
    s = ThermalSolver(mesh, T_init, back=back, evaporation=evap)
    M0 = float(np.sum(s.m))
    dev = {"max": 0.0}
    rec = lambda sv: dev.__setitem__("max", max(dev["max"], float(np.abs(sv.T - T_init).max())))
    n_steps, t_end = 10000, 1.0
    dt = t_end / n_steps
    s.run(t_end, dt0=dt, dt_max=dt, dT_target=BIG, dt_rel=BIG, record=rec)
    T = s.T
    Lg = s.ledger
    res_abs, _ = Lg.residual()
    out = dict(n=len(s.m), steps=s.n_steps_accepted, rejected=s.n_steps_rejected, evaporation_active=s.evaporation,
               max_dev_during_run_K=dev["max"], final_max_dev_K=float(np.abs(T - T_init).max()),
               surface_T_dev_K=s.surface_temperature() - T_init, mean_T_dev_K=float(np.sum(s.m * (T - T_init)) / np.sum(s.m)),
               dE_J_m2=Lg.stored_final - Lg.stored_initial, to_coolant_J_m2=Lg.to_coolant, vaporized_J_m2=Lg.vaporized,
               mass_evaporated_kg_m2=s.mass_evaporated, ledger_residual_J_m2=res_abs)
    assert s.n_steps_accepted >= n_steps and s.n_steps_rejected == 0
    if not s.evaporation:
        say(f"5_{mat_name}_{int(T_init)}K_{back_kind}_evapOff", **out)
        assert dev["max"] < 1e-12 and out["final_max_dev_K"] < 1e-12
        assert s.mass_evaporated == 0.0 and Lg.vaporized == 0.0 and abs(Lg.to_coolant) < 1e-9
        assert abs(out["dE_J_m2"]) < 1e-9
        return
    vap = mat.vapor
    j0 = float(vap.hkl_flux(T_init, s.alpha_evap))
    Lv = float(vap.L_v(T_init))
    h0 = float(mat.h_of_T(np.array([T_init]))[0])
    T_pred = float(mat.T_of_h(np.array([h0 - Lg.vaporized_latent / (M0 - s.mass_evaporated)]))[0])
    out.update(p_sat_Pa=float(vap.p_sat(T_init)), j0_kg_m2_s=j0, j0_Lv_t_J_m2=j0 * Lv * t_end, mass_over_j0t=s.mass_evaporated / (j0 * t_end),
               predicted_mean_dT_K=T_pred - T_init, closure_over_vaporized=res_abs / Lg.vaporized)
    say(f"5_{mat_name}_{int(T_init)}K_{back_kind}_evapOn", **out)
    assert abs(res_abs) < 1e-6 * Lg.vaporized + 1e-12
    if T_init < 2000.0:
        assert dev["max"] < 1e-12 and out["final_max_dev_K"] < 1e-12
        assert abs(s.mass_evaporated / (j0 * t_end) - 1) < 1e-9
    else:
        drift = T_pred - T_init
        assert drift < -0.05  # a measurable, physical drift
        assert abs(out["mean_T_dev_K"] - drift) < 1e-3 * abs(drift)
        assert abs(s.mass_evaporated / (j0 * t_end) - 1) < 2e-3
        assert abs(out["surface_T_dev_K"] - drift) < 1e-2 * abs(drift)
