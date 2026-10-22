"""Verification test 7: the Level-2 switch (He self-trapping + trap mutation) really changes the equations, and
why its effect in the baseline is small (about +2e-4 of the retained He).

Level 2 = Params.level2 = constants.level2_params(): He_2..He_7 clusters (J = 6) on top of the Level-1 species.
The Level-2 reactions are not written out in the docstrings of model.py / kernels.py (model.HeModel, the NumPy
reference of test3b, is Level 1 only), so they are derived here from the model description (Faney2013-type
diffusion-limited rates k = 4 pi (rA + rB)(DA + DB) N_C, immobile clusters) and implemented a second time in plain NumPy
(`l2_analytic`).  With r_He1 = 3e-10 m, r_c = R_C, D_s = D0_s exp(-Em_s/kT), N_C = N_W/1e6 and concentrations in appm:

  dimerisation   He + He -> He_2           rate 0.5 k11 He^2,  k11 = 4 pi (2 r_He1)(2 D_He) N_C   (2 He consumed)
  breakup        He_2 -> 2 He ; He_m -> He_(m-1) + He (m >= 3)   nu_HH exp(-(EbC_m + Em_He)/kT) C_m
  growth         He_m + He -> He_(m+1)     4 pi rCl_m D_He N_C He C_m           (m = 2 .. J)
  vacancy capt.  He_m + V -> HV_m          4 pi r_c D_V N_C V C_m               (m = 2 .. K)
  mutation       He_m -> HV_m + I          nu_tm,m exp(-Etm_m/kT) C_m           (m = 5, 6, 7)

(1) Unit level.  FastHe with level2 off and on (same Level-1 parameters), rhs of both compared at fixed states
    (T = 1000 K and a T-profile): rhs(L2) - rhs(L1) must (a) vanish exactly outside He, V, I, HV_2..HV_7, He_2..He_7 and
    the ledger header, (b) equal the analytic sum of the new terms, term by term (each term isolated by choosing the
    state/parameters so that only it is non-zero), (c) conserve He (weights 1, k, m) with the Jacobian included
    (c^T J = 0), and the Jacobian must match finite differences.
(2) Forced-on runs: constant T, xi = 0, no neutrons, no free vacancies at all, He implanted at he_scale x the source
    (Prototype ion spectrum, one 1-s cycle per shot).  Level 1 has nothing to trap the He (it only diffuses; ~1 % is
    still mobile deep in the wall after 1 s); Level 2 self-traps, grows clusters and creates vacancies by trap mutation.
    Second demonstration: Demo cycle, two shots, he_scale 1 ... 1e3.
(3) The baseline (he_scale = 1): where Level 2 acts inside one Demo cycle and why the net effect is only ~1e-4: depth-resolved
    mobile He and free V in the final Demo baseline state (end of cycle and maximum over the cycle), the ratio self-trapping /
    vacancy-trapping rate, the Level-2 reaction budget of one cycle, the exact L2-vs-L1 numbers of the stored 400-shot runs,
    and a one-shot test from the SAME state at several solver tolerances (is the residual difference numerical or physical?).

Pass: forced-on retained He differs by > 5 %  AND  He conservation < 1e-10 in every run  AND  the term-by-term check passes.
Usage: python scripts/verify_he_7_level2.py [p1] [p2] [p3] [fig]     (no argument = everything, records the result)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import warnings
from scipy.integrate import solve_ivp

from _he_common import ROOT, record, style, VER
from ife_wall.scenarios import converged_preshot  # noqa: E402
from ife_wall.he import constants as C
from ife_wall.he import kernels as kn
from ife_wall.he.fast import FastHe
from ife_wall.he.grid import make_grid
from ife_wall.he.model import NC, CycleTemperature
from ife_wall.he.simulate import FastSimulation
from ife_wall.he.sources import ShotSources

KB = 8.617333262e-5
RUNS = ROOT / "runs" / "he_retention"
CYC = str(RUNS / "thermal" / f"Demo_T{converged_preshot('Demo', legacy=921.1):.1f}_ss.npz")
GROUPS = ("dimerisation", "breakup", "growth", "vacancy_capture", "mutation")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def tolist(x):
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ======================================================================================================
# (1) unit level: independent NumPy implementation of the Level-2 terms
# ======================================================================================================
def labels_of(f):
    return ["He", "V", "I"] + [f"HV{k}" for k in range(1, f.K + 1)] + [f"C{m}" for m in range(2, f.J + 2)]


def l2_analytic(y, Tcell, f, l2):
    """Level-2-only contribution to dy/dt of the cell species (n, S) in appm/s, computed from the formulas in the module
    docstring (not from kernels.py).  Also returns the gross rate of each reaction group, (n,) arrays (appm/s)."""
    n, K, J = f.n, f.K, f.J
    u = y[kn.H:].reshape(n, f.S)
    kT = KB * Tcell
    DHe = C.D0_HE * np.exp(-C.EM_HE / kT)            # m^2/s
    DV = C.D0_V * np.exp(-C.EM_V / kT)
    k4 = 4.0 * np.pi * NC                            # (4 pi) x (appm -> m^-3)
    He, V = u[:, 0], u[:, 1]
    cl = lambda m: 3 + K + (m - 2)                   # species index of He_m
    hv = lambda k: 2 + k                             # species index of HV_k
    d = np.zeros((n, f.S))
    R = {g: np.zeros(n) for g in GROUPS}
    R["breakup_C2"] = np.zeros(n)                    # extra: He_2 -> 2 He events
    R["He_to_HV"] = np.zeros(n)                      # extra: He atoms moved from clusters into HV_m (vacancy capture + mutation)
    # dimerisation: He + He -> He_2
    k11 = k4 * (2.0 * l2["rHe1"]) * (2.0 * DHe)
    r = 0.5 * k11 * He ** 2
    d[:, 0] -= 2.0 * r
    d[:, cl(2)] += r
    R["dimerisation"] += r
    for s in range(2, J + 2):                        # breakup of He_s
        r = l2["nuHeHe"] * np.exp(-(l2["EbC"][s - 2] + C.EM_HE) / kT) * u[:, cl(s)]
        d[:, cl(s)] -= r
        if s == 2:
            d[:, 0] += 2.0 * r
            R["breakup_C2"] += r
        else:
            d[:, cl(s - 1)] += r
            d[:, 0] += r
        R["breakup"] += r
    for m in range(2, J + 1):                        # growth He_m + He -> He_(m+1)
        r = k4 * l2["rCl"][m - 2] * DHe * He * u[:, cl(m)]
        d[:, cl(m)] -= r
        d[:, 0] -= r
        d[:, cl(m + 1)] += r
        R["growth"] += r
    for m in range(2, J + 2):
        if m <= K:                                   # vacancy capture He_m + V -> HV_m
            r = k4 * C.R_C * DV * V * u[:, cl(m)]
            d[:, cl(m)] -= r
            d[:, 1] -= r
            d[:, hv(m)] += r
            R["vacancy_capture"] += r
            R["He_to_HV"] += m * r
        if l2["nutm"][m - 2] > 0 and m <= K:         # trap mutation He_m -> HV_m + I
            r = l2["nutm"][m - 2] * np.exp(-l2["Etm"][m - 2] / kT) * u[:, cl(m)]
            d[:, cl(m)] -= r
            d[:, hv(m)] += r
            d[:, 2] += r
            R["mutation"] += r
            R["He_to_HV"] += m * r
    return d, R


def he_weights(f):
    return np.r_[1.0, 0.0, 0.0, np.arange(1, f.K + 1), np.arange(2, f.J + 2)]


def cvec(f):
    """He-conservation covector: surface-loss ledger (+1), implanted ledger (-1), cell He (mobile 1, HV_k k, He_m m) x dx."""
    c = np.zeros(f.N)
    c[0], c[3] = 1.0, -1.0
    c[kn.H:] = (f.dx[:, None] * he_weights(f)[None, :]).ravel()
    return c


def idx_map(f1, f2):
    idx = np.empty(f1.N, int)
    idx[:kn.H] = np.arange(kn.H)
    for i in range(f1.n):
        for s in range(f1.S):
            idx[kn.H + i * f1.S + s] = kn.H + i * f2.S + s
    return idx


def to_L1(y2, f1, f2):
    y1 = np.zeros(f1.N)
    y1[:kn.H] = y2[:kn.H]
    y1[kn.H:] = y2[kn.H:].reshape(f2.n, f2.S)[:, :f1.S].ravel()
    return y1


def make_state(f2, He=0.0, V=0.0, I=0.0, HV=0.0, Cl=0.0):
    n, K, J = f2.n, f2.K, f2.J
    u = np.zeros((n, f2.S))
    u[:, 0] = He
    u[:, 1] = V
    u[:, 2] = I
    u[:, 3:3 + K] = HV
    u[:, 3 + K:3 + K + J] = Cl
    y = np.zeros(f2.N)
    y[kn.H:] = u.ravel()
    return y


def lu(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def setup(f, Tcell, bgHe, bgV):
    f.set_cycle([0.0, 1e30], np.vstack([Tcell, Tcell]))
    f.bgHe[:] = bgHe
    f.bgV[:] = bgV


def delta_case(name, f1, f2, l2m, y2, Tcell, t=0.3):
    """kernel Delta = rhs(L2) - rhs(L1, same state without clusters) against the analytic Level-2 terms."""
    idx = idx_map(f1, f2)
    y1 = to_L1(y2, f1, f2)
    r2, r1 = f2.rhs(t, y2), f1.rhs(t, y1)
    D = r2.copy()
    D[idx] -= r1
    d, R = l2_analytic(y2, Tcell, f2, l2m)
    Da = np.zeros(f2.N)
    Da[kn.H:] = d.ravel()
    lab = labels_of(f2)
    uk, ua = D[kn.H:].reshape(f2.n, f2.S), Da[kn.H:].reshape(f2.n, f2.S)
    t_k = [lab[s] for s in range(f2.S) if np.any(uk[:, s] != 0.0)]
    t_a = [lab[s] for s in range(f2.S) if np.any(ua[:, s] != 0.0)]
    c2 = cvec(f2)
    out = dict(case=name, short=name.split(" (")[0].split(",")[0], rel_err=float(np.abs(D - Da).max() / np.abs(Da).max()),
               touched_kernel=t_k, touched_analytic=t_a, touched_equal=(t_k == t_a),
               header_delta=float(np.abs(D[:kn.H]).max()), HV1_delta=float(np.abs(uk[:, 3]).max()),
               active_groups=[g for g in GROUPS if R[g].max() > 0.0],
               max_abs_delta=float(np.abs(D).max()), max_abs_rhs_L1=float(np.abs(r1).max()),
               He_cons_of_delta=float(abs(c2 @ D) / (np.abs(c2) * np.abs(D)).sum()),
               He_cons_of_analytic=float(abs(c2 @ Da) / (np.abs(c2) * np.abs(Da)).sum()),
               rates_max={g: float(R[g].max()) for g in GROUPS})
    return out


def part1():
    log("(1) unit level: Level-1 vs Level-2 right-hand sides")
    l2 = C.level2_params()
    rng = np.random.default_rng(20261001)
    n = 24
    edges = np.r_[0.0, np.cumsum(rng.uniform(2.0, 6.0, n))] * 1e-6          # 24 cells of 2-6 um (diffusion ~ reaction time scales)
    f1 = FastHe(edges, C.baseline())
    f2 = FastHe(edges, C.baseline(level2=l2))
    out = dict(S_L1=f1.S, S_L2=f2.S, N_L1=f1.N, N_L2=f2.N, J=f2.J)
    bgHe, bgV = lu(rng, 1e-2, 1e1, n), lu(rng, 1e-2, 1e1, n)
    Tuni, Tvar = np.full(n, 1000.0), np.linspace(700.0, 1500.0, n)
    cases = []
    # states
    HeA, VA, IA, HVA, CA = lu(rng, 10, 1e3, n), lu(rng, 1, 100, n), lu(rng, 1e-3, 1e-1, n), lu(rng, .1, 10, (n, f2.K)), lu(rng, .1, 10, (n, f2.J))
    HeB = lu(rng, 1e2, 1e4, n)
    for Tname, Tc in (("T=1000 K", Tuni), ("T=700..1500 K", Tvar)):
        for f in (f1, f2):
            setup(f, Tc, bgHe, bgV)
        cases.append(delta_case(f"all terms, mobile He + V + clusters, {Tname}", f1, f2, l2, make_state(f2, HeA, VA, IA, HVA, CA), Tc))
        cases.append(delta_case(f"large He, NO vacancies, no clusters, {Tname}", f1, f2, l2, make_state(f2, HeB), Tc))
        cases.append(delta_case(f"large He, NO vacancies, with clusters, {Tname}", f1, f2, l2, make_state(f2, HeB, Cl=CA), Tc))
    # isolated terms (T = 1000 K): the other terms are zero because of the state and/or the parameters
    for f in (f1,):
        setup(f, Tuni, bgHe, bgV)
    iso = [("dimerisation only (clusters = 0)", dict(), make_state(f2, HeA, VA, IA, HVA, 0.0)),
           ("breakup only (He = V = I = HV = 0, nu_tm = 0)", dict(nutm=np.zeros(f2.J)), make_state(f2, 0.0, 0.0, 0.0, 0.0, CA)),
           ("growth only (V = I = HV = 0, r_He1 = 0, nu_HH = 0, nu_tm = 0)", dict(rHe1=0.0, nuHeHe=0.0, nutm=np.zeros(f2.J)), make_state(f2, HeA, 0.0, 0.0, 0.0, CA)),
           ("vacancy capture only (He = I = HV = 0, nu_HH = 0, nu_tm = 0)", dict(nuHeHe=0.0, nutm=np.zeros(f2.J)), make_state(f2, 0.0, VA, 0.0, 0.0, CA)),
           ("trap mutation only (He = V = I = HV = 0, nu_HH = 0)", dict(nuHeHe=0.0), make_state(f2, 0.0, 0.0, 0.0, 0.0, CA))]
    for name, mod, y2 in iso:
        l2m = dict(l2, **mod)
        f2m = FastHe(edges, C.baseline(level2=l2m))
        setup(f2m, Tuni, bgHe, bgV)
        cases.append(delta_case(name, f1, f2m, l2m, y2, Tuni))
    out["cases"] = cases
    for cs in cases:
        log(f"   {cs['case']:70s} rel err {cs['rel_err']:.1e}  touched {cs['touched_kernel']}  groups {cs['active_groups']}")
    # He conservation of rhs and Jacobian, both levels; production grid and the small grid
    cons = {}
    for f_, nm in ((f1, "L1"), (f2, "L2")):
        setup(f_, Tuni, bgHe, bgV)
    yA2 = make_state(f2, HeA, VA, IA, HVA, CA)
    yA1 = to_L1(yA2, f1, f2)
    t = 0.3
    for f_, y_, nm in ((f1, yA1, "L1"), (f2, yA2, "L2")):
        c = cvec(f_)
        fr = f_.rhs(t, y_)
        J = f_.dense_jac(t, y_)
        cJ = c @ J
        sc = (np.abs(c)[:, None] * np.abs(J)).sum(0)
        cons[f"{nm}_rhs"] = float(abs(c @ fr) / (np.abs(c) * np.abs(fr)).sum())
        cons[f"{nm}_cT_J"] = float((np.abs(cJ) / np.maximum(sc, 1e-300)).max())
        if nm == "L2":
            J2 = J
        else:
            J1 = J
    idx = idx_map(f1, f2)
    J1e = np.zeros_like(J2)
    J1e[np.ix_(idx, idx)] = J1
    dJ = J2 - J1e
    c2 = cvec(f2)
    cons["L2_minus_L1_cT_dJ"] = float((np.abs(c2 @ dJ) / np.maximum((np.abs(c2)[:, None] * np.abs(dJ)).sum(0), 1e-300)).max())
    out["nnz_dJ"] = int((dJ != 0).sum())
    # Jacobian against finite differences (rhs is quadratic in y, so central differences are exact up to round-off)
    ref = np.abs(J2).max()
    fd_err = 0.0
    for j in range(f2.N):
        h = 1e-3 * max(abs(yA2[j]), 1e-3)
        yp, ym = yA2.copy(), yA2.copy()
        yp[j] += h
        ym[j] -= h
        col = (f2.rhs(t, yp) - f2.rhs(t, ym)) / (2 * h)
        fd_err = max(fd_err, np.abs(col - J2[:, j]).max() / max(np.abs(J2[:, j]).max(), 1e-6 * ref))
    cons["L2_jacobian_vs_finite_differences"] = float(fd_err)
    # production grid, rhs and banded c^T J
    pe = make_grid()
    g1, g2 = FastHe(pe, C.baseline()), FastHe(pe, C.baseline(level2=l2))
    m = len(pe) - 1
    for g in (g1, g2):
        setup(g, np.full(m, 1000.0), lu(rng, 1e-2, 1e1, m), lu(rng, 1e-2, 1e1, m))
    y2p = make_state(g2, lu(rng, 10, 1e3, m), lu(rng, 1, 100, m), lu(rng, 1e-3, 1e-1, m), lu(rng, .1, 10, (m, g2.K)), lu(rng, .1, 10, (m, g2.J)))
    c = cvec(g2)
    fr = g2.rhs(t, y2p)
    cons["production_grid_L2_rhs"] = float(abs(c @ fr) / (np.abs(c) * np.abs(fr)).sum())
    ab = g2.jac(t, y2p)
    worst = 0.0
    for j in range(g2.N):
        rows = np.arange(max(0, j - g2.mu), min(g2.N, j + g2.mu + 1))
        col = ab[g2.mu + rows - j, j]
        worst = max(worst, abs((c[rows] * col).sum()) / max((np.abs(c[rows]) * np.abs(col)).sum(), 1e-300))
    cons["production_grid_L2_cT_J"] = float(worst)
    out["conservation"] = cons
    log("   conservation (relative to sum of |terms|):", {k: f"{v:.1e}" for k, v in cons.items()})
    out["max_rel_err_terms"] = max(cs["rel_err"] for cs in cases)
    out["untouched_ok"] = all(cs["header_delta"] == 0.0 and cs["HV1_delta"] == 0.0 and cs["touched_equal"] for cs in cases)
    out["analytic_He_conserving"] = max(cs["He_cons_of_analytic"] for cs in cases)
    out["max_cons"] = max(v for k, v in cons.items() if "finite" not in k)
    out["passed"] = bool(out["max_rel_err_terms"] < 1e-9 and out["untouched_ok"] and out["max_cons"] < 1e-12
                         and cons["L2_jacobian_vs_finite_differences"] < 1e-6)
    log(f"   term-by-term: max rel err {out['max_rel_err_terms']:.1e}, untouched components exactly zero & touched sets equal: "
        f"{out['untouched_ok']}, max conservation residual {out['max_cons']:.1e}, FD Jacobian {fd_err:.1e}  -> passed {out['passed']}")
    return out


# ======================================================================================================
# (2) forced-on runs
# ======================================================================================================
L1P = C.baseline()
L2P = C.baseline(level2=C.level2_params())


def derived(sn):
    sn = dict(sn)
    sn["V_net"] = sn["V_free"] + sn["V_He"]
    if "He_cons_err" in sn:
        sn["cons_rel"] = abs(sn["He_cons_err"]) / sn["He_in"]
    return sn


def forced_shots(src, p, T, hs, xi, n_shots, rtol, atol):
    sim = FastSimulation("Prototype", p, None, sources=src, T_const=T, P=1.0, xi=xi, neutron=False, he_scale=hs, rtol=rtol, atol=atol)
    rows = []
    for _ in range(n_shots):
        sim.shot()
        rows.append(derived(sim.snapshot()))
    return rows


def forced_time(src, p, T, hs, xi=0.0, rtol=1e-6, atol=1e-12):
    """shot 1 integrated through the ion-arrival slices and a log-spaced cooling tail; totals at every node."""
    sim = FastSimulation("Prototype", p, None, sources=src, T_const=T, P=1.0, xi=xi, neutron=False, he_scale=hs, rtol=rtol, atol=atol)
    te = src.t_edges
    nodes = np.unique(np.r_[0.0, te, np.geomspace(2 * te[-1], 1.0, 40), 1.0])
    y = sim.y
    rec = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        y = sim.fm.integrate(y, a, b, rtol=rtol, atol=atol)
        sn = derived(sim.fm.totals(y))
        sn["He_retained"] = sn["He_mobile"] + sn["He_trapped"] + sn["He_clusters"]
        sn["t"] = b
        rec.append(sn)
    return rec


def part2(src_p, src_d, quick=False):
    log("(2) forced-on runs (Prototype ion spectrum, constant T, xi = 0, no neutrons, P = 1 s)")
    out = dict(rtol=1e-6, atol=1e-12)
    table = []
    cases = [(1000.0, 1.0), (1000.0, 1e2), (1000.0, 1e4), (1500.0, 1.0), (1500.0, 1e2), (1500.0, 1e4)]
    if quick:
        cases = [(1000.0, 1e2)]
    for T, hs in cases:
        a, b = (forced_shots(src_p, p, T, hs, 0.0, 1, 1e-6, 1e-12)[0] for p in (L1P, L2P))
        row = dict(T=T, he_scale=hs, xi=0.0, He_in=a["He_in"], L1=a, L2=b,
                   rel_diff=(b["He_retained"] - a["He_retained"]) / a["He_retained"],
                   retained_frac_L1=a["He_retained"] / a["He_in"], retained_frac_L2=b["He_retained"] / b["He_in"])
        table.append(row)
        log(f"   T={T:.0f} K he_scale={hs:g}: retained L1 {a['He_retained']:.4e} ({100 * row['retained_frac_L1']:.2f} %)  "
            f"L2 {b['He_retained']:.4e} ({100 * row['retained_frac_L2']:.2f} %)  x{b['He_retained'] / a['He_retained']:.2f}; "
            f"He in clusters {b['He_clusters']:.3e}; vacancies L1 {a['V_net']:.2e} L2 {b['V_net']:.3e}; "
            f"cons L1 {a['cons_rel']:.1e} L2 {b['cons_rel']:.1e}")
    out["table"] = table
    # tiny vacancy source instead of xi = 0 (Level 1 then also has traps)
    xi_tiny = None
    if not quick:
        a, b = (forced_shots(src_p, p, 1000.0, 1e2, 1e-4, 1, 1e-6, 1e-12)[0] for p in (L1P, L2P))
        xi_tiny = dict(T=1000.0, he_scale=1e2, xi=1e-4, L1=a, L2=b, rel_diff=(b["He_retained"] - a["He_retained"]) / a["He_retained"])
        log(f"   xi = 1e-4: retained L1 {a['He_retained']:.4e} (trapped {a['He_trapped']:.2e}), L2 {b['He_retained']:.4e}, rel diff {xi_tiny['rel_diff']:.2f}")
    out["xi_tiny"] = xi_tiny
    # headline: T = 1000 K, he_scale = 100, three shots
    n_sh = 1 if quick else 3
    sh1 = forced_shots(src_p, L1P, 1000.0, 1e2, 0.0, n_sh, 1e-6, 1e-12)
    sh2 = forced_shots(src_p, L2P, 1000.0, 1e2, 0.0, n_sh, 1e-6, 1e-12)
    out["headline"] = dict(T=1000.0, he_scale=1e2, xi=0.0, shots=n_sh, L1=sh1, L2=sh2)
    for k, (a, b) in enumerate(zip(sh1, sh2), 1):
        log(f"   headline shot {k}: retained L1 {a['He_retained']:.4e}  L2 {b['He_retained']:.4e}  clusters {b['He_clusters']:.3e}  "
            f"V_net L2 {b['V_net']:.3e}  cons {a['cons_rel']:.1e}/{b['cons_rel']:.1e}")
    h1, h2 = sh1[0], sh2[0]
    out["headline_rel_diff"] = float((h2["He_retained"] - h1["He_retained"]) / h1["He_retained"])
    out["headline_frac_diff"] = float((h2["He_retained"] - h1["He_retained"]) / h1["He_in"])
    # tolerance check of the headline case
    if not quick:
        t1, t2 = (forced_shots(src_p, p, 1000.0, 1e2, 0.0, 1, 1e-8, 1e-14)[0] for p in (L1P, L2P))
        out["tight_tol"] = dict(rtol=1e-8, atol=1e-14, L1_retained=t1["He_retained"], L2_retained=t2["He_retained"],
                                L1_change=t1["He_retained"] / h1["He_retained"] - 1, L2_change=t2["He_retained"] / h2["He_retained"] - 1,
                                L2_clusters=t2["He_clusters"], L2_V_net=t2["V_net"])
        log(f"   headline at rtol 1e-8: L1 {t1['He_retained']:.4e} ({out['tight_tol']['L1_change']:+.1e}), L2 {t2['He_retained']:.4e} "
            f"({out['tight_tol']['L2_change']:+.1e})")
    out["time"] = dict(L1=forced_time(src_p, L1P, 1000.0, 1e2), L2=forced_time(src_p, L2P, 1000.0, 1e2))
    out["max_cons"] = max([r["cons_rel"] for row in table for r in (row["L1"], row["L2"])] + [r["cons_rel"] for r in sh1 + sh2]
                          + ([xi_tiny["L1"]["cons_rel"], xi_tiny["L2"]["cons_rel"]] if xi_tiny else []))
    # second demonstration: Demo cycle, two shots, he_scale scan
    log("(2b) Demo cycle (xi = 1, neutrons on), 2 shots from an empty wall, he_scale scan")
    scan = []
    for hs in ([1.0, 1e3] if quick else [1.0, 10.0, 100.0, 1e3]):
        res = []
        for p in (L1P, L2P):
            sim = FastSimulation("Demo", p, CYC, sources=src_d, he_scale=hs)
            for _ in range(2):
                sim.shot()
            res.append(derived(sim.snapshot()))
        a, b = res
        row = dict(he_scale=hs, L1=a, L2=b, rel_diff=b["He_retained"] / a["He_retained"] - 1)
        scan.append(row)
        log(f"   he_scale={hs:g}: retained L1 {a['He_retained']:.5e} L2 {b['He_retained']:.5e} -> {100 * row['rel_diff']:+.4g} %  "
            f"(L2: clusters {b['He_clusters']:.2e}, V_He {b['V_He']:.3e} vs L1 {a['V_He']:.3e}; cons {a['cons_rel']:.1e}/{b['cons_rel']:.1e})")
    out["demo_scan"] = scan
    out["max_cons"] = max([out["max_cons"]] + [r["cons_rel"] for row in scan for r in (row["L1"], row["L2"])])
    return out


# ======================================================================================================
# (3) baseline: Level 2 inside one cycle, and why the net effect is small
# ======================================================================================================
def rate_fields(y, T, f, l2=None):
    """per-cell rates (appm/s) at state y: He consumed by self-trapping (dimerisation k11 He^2, plus cluster growth if the
    state has clusters) and by trapping at vacancies / He-V (sum_k a_k He V_k), plus the cluster-channel groups."""
    n, K = f.n, f.K
    u = y[kn.H:].reshape(n, f.S)
    kT = KB * T
    DHe, DV = C.D0_HE * np.exp(-C.EM_HE / kT), C.D0_V * np.exp(-C.EM_V / kT)
    k4 = 4.0 * np.pi * NC
    He = u[:, 0]
    Vk = np.zeros((n, K))
    Vk[:, 0] = u[:, 1]
    Vk[:, 1:] = u[:, 3:3 + K - 1]                       # V_0 = V, V_k = HV_k (k = 1 .. K-1 can still take a He)
    a = np.empty((n, K))
    a[:, 0] = k4 * C.R_C * (DHe + DV)
    a[:, 1:] = (k4 * C.R_C * DHe)[:, None]
    Rvac = (a * Vk).sum(1) * He
    Rdim = k4 * (2 * C.R_HE1) * (2 * DHe) * He ** 2     # = 2 x (0.5 k11 He^2): He atoms consumed per appm/s
    HV = u[:, 3:3 + K]
    b = C.NU * np.exp(-(C.EB_BASE[None, :] + C.EM_HE) / kT[:, None])
    e_ann = k4 * C.R_IV * C.D0_I * np.exp(-C.EM_I / kT)
    Rdetrap = (b * HV).sum(1)                                           # He released by thermal detrapping of HV_k
    Rann = e_ann * u[:, 2] * (HV * np.arange(1, K + 1)[None, :]).sum(1)  # He released by I + HV_k -> k He
    d = dict(He=He, V=u[:, 1], Vk=Vk, Rvac=Rvac, Rdim=Rdim, Rself=Rdim.copy(), Rdetrap=Rdetrap, Rann=Rann, kappa=float(4 * C.R_HE1 / C.R_C))
    if f.J and l2 is not None:
        _, R = l2_analytic(y, T, f, l2)
        d["groups"] = R
        d["Rself"] = Rdim + R["growth"]
    return d


def dense_cycle(src, p, y0, l2=None, n_win=300, n_tail=150):
    """One Demo cycle (production settings rtol 1e-3, atol 1e-9) started from the stored state y0 and sampled on ~450 times
    (solve_ivp t_eval): 300 log-spaced inside the ion-arrival window, 150 in the cooling tail.  Returns rates and totals."""
    sim = FastSimulation("Demo", p, CYC, sources=src)
    fm = sim.fm
    cyc = CycleTemperature(CYC, fm.x)
    te, P = src.t_edges, sim.P
    pieces = [(0.0, te[0], np.geomspace(1e-9, te[0], 6)), (te[0], te[-1], np.geomspace(te[0], te[-1], n_win)),
              (te[-1], P, np.geomspace(te[-1], P, n_tail))]
    y = y0.copy()
    ts, ys = [0.0], [y.copy()]
    at = np.full(fm.N, sim.atol)
    at[:kn.H] = 1e-9
    for a, b, tg in pieces:
        tg = tg[(tg > a) & (tg <= b)]
        if tg[-1] < b:
            tg = np.r_[tg, b]
        sol = solve_ivp(fm.rhs, (a, b), y, method="LSODA", jac=fm.jac, lband=fm.mu, uband=fm.mu, rtol=sim.rtol, atol=at,
                        t_eval=tg, first_step=(b - a) * 1e-6)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts += list(sol.t)
        ys += [sol.y[:, k].copy() for k in range(sol.y.shape[1])]
        y = sol.y[:, -1].copy()
    ts = np.array(ts)
    Tt = np.array([cyc(t) for t in ts])
    fields = [rate_fields(yy, Tt[i], fm, l2) for i, yy in enumerate(ys)]
    Y = np.array(ys)
    u = Y[:, kn.H:].reshape(len(ts), fm.n, fm.S)
    w = fm.dx * 1e-6 * NC
    K, J = fm.K, fm.J
    return dict(t=ts, fields=fields, dx=fm.dx, x=fm.x * 1e6, w=w, P=P, y_end=y, fm=fm, T_min=float(Tt.min()), T_max=float(Tt.max()),
                He_cl_cell=((u[:, :, 3 + K:3 + K + J] * np.arange(2, J + 2)[None, None, :]).sum(2) if J else np.zeros((len(ts), fm.n))),
                He_mobile=(u[:, :, 0] * w).sum(1),
                He_trapped=((u[:, :, 3:3 + K] * np.arange(1, K + 1)[None, None, :]).sum(2) * w).sum(1),
                He_clusters=(((u[:, :, 3 + K:3 + K + J] * np.arange(2, J + 2)[None, None, :]).sum(2) * w).sum(1) if J else np.zeros(len(ts))),
                V_net=((u[:, :, 1] + u[:, :, 3:3 + K].sum(2)) * w).sum(1),
                He_surface=Y[:, 0] * 1e-6 * NC, He_in=Y[:, 3] * 1e-6 * NC, V_prod=Y[:, 4] * 1e-6 * NC)


def tint(t, A, dx):
    """time integral of the depth integral of a per-cell rate A[t, x] (appm/s): appm um/s -> per m^2 per cycle."""
    return float(np.trapezoid((A * dx).sum(1), t)) * 1e-6 * NC


def cycle_stats(dc):
    F, dx, t, x = dc["fields"], dc["dx"], dc["t"], dc["x"]
    Rs = np.array([f["Rself"] for f in F])
    Rv = np.array([f["Rvac"] for f in F])
    He = np.array([f["He"] for f in F])
    Vf = np.array([f["V"] for f in F])
    Vsum = np.array([f["Vk"].sum(1) for f in F])
    ratio_int = (Rs * dx).sum(1) / np.maximum((Rv * dx).sum(1), 1e-300)
    cell = np.where(Rv > 0, Rs / np.where(Rv > 0, Rv, 1.0), np.nan)
    rel = Rv >= 1e-3 * Rv.max(1, keepdims=True)                      # cells that carry the vacancy trapping at that instant
    cr = np.where(rel, np.nan_to_num(cell), 0.0)
    k, j = np.unravel_index(np.argmax(cr), cr.shape)
    gt1 = t[ratio_int > 1.0]
    st = dict(He_mobile_total_peak=float(dc["He_mobile"].max()), t_He_mobile_total_peak=float(t[np.argmax(dc["He_mobile"])]),
              He_mobile_total_start=float(dc["He_mobile"][0]), He_mobile_total_end=float(dc["He_mobile"][-1]),
              He_cell_max_appm=float(He.max()), depth_of_He_cell_max_um=float(x[np.unravel_index(np.argmax(He), He.shape)[1]]),
              t_of_He_cell_max=float(t[np.unravel_index(np.argmax(He), He.shape)[0]]),
              ratio_int_start=float(ratio_int[0]), ratio_int_max=float(ratio_int.max()), t_ratio_int_max=float(t[np.argmax(ratio_int)]),
              ratio_int_gt1_t_from=float(gt1.min()) if len(gt1) else None, ratio_int_gt1_t_to=float(gt1.max()) if len(gt1) else None,
              ratio_cell_max=float(cr[k, j]), depth_of_ratio_cell_max_um=float(x[j]), t_of_ratio_cell_max=float(t[k]),
              He_at_ratio_cell_max_appm=float(He[k, j]), Vsum_at_ratio_cell_max_appm=float(Vsum[k, j]),
              int_self_trapping_He_per_cycle=tint(t, Rs, dx), int_vacancy_trapping_He_per_cycle=tint(t, Rv, dx))
    st["He_released_by_thermal_detrapping_per_cycle"] = tint(t, np.array([f["Rdetrap"] for f in F]), dx)
    st["He_released_by_I_HV_annihilation_per_cycle"] = tint(t, np.array([f["Rann"] for f in F]), dx)
    st["T_min"], st["T_max"] = dc["T_min"], dc["T_max"]
    st["int_ratio_cycle"] = st["int_self_trapping_He_per_cycle"] / st["int_vacancy_trapping_He_per_cycle"]
    st["He_cell_max_free_V_appm"] = float(Vf[np.unravel_index(np.argmax(He), He.shape)])
    arr = dict(Rs=Rs, Rv=Rv, He=He, Vf=Vf, Vsum=Vsum, cell=cell, ratio_int=ratio_int)
    return st, arr


def one_shot_same_state(src, y_L2, f1, f2, rtols):
    """Start both models from the same state (L2 state; the clusters are dropped for L1), advance one shot at several
    tolerances and compare the increments of retained He / released He."""
    y_L1 = to_L1(y_L2, f1, f2)
    res = {}
    for rt, at in rtols:
        r = {}
        for lab, p, yy in (("L1", L1P, y_L1), ("L2", L2P, y_L2)):
            sim = FastSimulation("Demo", p, CYC, sources=src, rtol=rt, atol=at)
            sim.y = yy.copy()
            s0 = derived(sim.snapshot())
            t0 = time.time()
            sim.shot()
            s1 = derived(sim.snapshot())
            r[lab] = dict(dR=s1["He_retained"] - s0["He_retained"], dSurf=s1["He_surface"] - s0["He_surface"], dHe_in=s1["He_in"] - s0["He_in"],
                          R0=s0["He_retained"], clusters_end=s1["He_clusters"], wall=time.time() - t0)
        a, b = r["L1"], r["L2"]
        r["rel_increment"] = (b["dR"] - a["dR"]) / a["dR"]
        r["rel_release"] = (b["dSurf"] - a["dSurf"]) / a["dSurf"]
        r["diff_over_R0"] = (b["dR"] - a["dR"]) / b["R0"]
        r["dHe_in_rel"] = b["dHe_in"] / a["dHe_in"] - 1
        res[f"{rt:g}"] = r
        log(f"   same-state one shot, rtol {rt:g}: dR L1 {a['dR']:.6e}  L2 {b['dR']:.6e}  -> (L2-L1)/dR = {r['rel_increment']:+.3e}; release {r['rel_release']:+.3e}; "
            f"(L2-L1)/R0 = {r['diff_over_R0']:+.3e}; source ledger differs by {r['dHe_in_rel']:+.1e}  ({a['wall']:.0f}+{b['wall']:.0f} s)")
    return res


def part3(src_d, quick=False):
    log("(3) baseline: Level 2 inside one Demo cycle, and why the net effect is small")
    out = {}
    e = make_grid()
    l2 = C.level2_params()
    # --- stored runs ------------------------------------------------------------------------------
    a = np.genfromtxt(RUNS / "Demo_baseline" / "totals.csv", delimiter=",", names=True)
    b = np.genfromtxt(RUNS / "sens" / "Demo_level2" / "totals.csv", delimiter=",", names=True)
    meta2 = json.loads((RUNS / "sens" / "Demo_level2" / "meta.json").read_text(encoding="utf-8"))
    meta1 = json.loads((RUNS / "Demo_baseline" / "meta.json").read_text(encoding="utf-8"))
    rows = {}
    for s in (1, 10, 30, 100, 200, 300, 400):
        i = s - 1
        assert a["shot"][i] == s and b["shot"][i] == s
        rows[s] = dict(L1=float(a["He_retained"][i]), L2=float(b["He_retained"][i]), rel=float(b["He_retained"][i] / a["He_retained"][i] - 1),
                       He_in_rel=float(b["He_in"][i] / a["He_in"][i] - 1), clusters=float(b["He_clusters"][i]))
    out["stored_runs"] = rows
    out["meta_L2_level2"] = dict(J=meta2["params"]["level2"]["J"], nuHeHe=meta2["params"]["level2"]["nuHeHe"], rHe1=meta2["params"]["level2"]["rHe1"],
                                 sim=meta2["sim"], shots=meta2["shots_done"])
    out["meta_L1_level2"] = meta1["params"]["level2"]
    out["L2_clusters_min_max"] = [float(b["He_clusters"].min()), float(b["He_clusters"].max())]
    out["L2_run_cons_max"] = float((np.abs(b["He_cons_err"]) / b["He_in"]).max())
    out["L1_run_cons_max"] = float((np.abs(a["He_cons_err"]) / a["He_in"]).max())
    for s, r in rows.items():
        log(f"   shot {s:3d}: L1 {r['L1']:.6e}  L2 {r['L2']:.6e}  L2/L1-1 = {r['rel']:+.4e}   (clusters {r['clusters']:.3e})")
    log(f"   meta: L2 level2 J = {out['meta_L2_level2']['J']}, nuHeHe = {out['meta_L2_level2']['nuHeHe']:g}, rHe1 = {out['meta_L2_level2']['rHe1']:g}; L1 level2 = {out['meta_L1_level2']}; "
        f"He_clusters over the 400 L2 shots in [{out['L2_clusters_min_max'][0]:.2e}, {out['L2_clusters_min_max'][1]:.2e}] He/m2 (> 0); "
        f"max |He_cons_err|/He_in: L2 {out['L2_run_cons_max']:.1e}, L1 {out['L1_run_cons_max']:.1e}")
    z1 = np.load(RUNS / "Demo_baseline" / "checkpoint.npz", allow_pickle=True)
    z2 = np.load(RUNS / "sens" / "Demo_level2" / "checkpoint.npz", allow_pickle=True)
    out["checkpoint_shots"] = [int(z1["shot_no"]), int(z2["shot_no"])]
    f1, f2 = FastHe(e, L1P), FastHe(e, L2P)
    pr = f1.profiles(z1["y"])
    # --- dense cycles: (A) final Demo baseline state (L1, 1450 shots), (B) the 400-shot state with L1, (C) the same with L2 ------------
    dcA = dense_cycle(src_d, L1P, z1["y"])
    dcB = dense_cycle(src_d, L1P, to_L1(z2["y"], f1, f2))
    dcC = dense_cycle(src_d, L2P, z2["y"], l2)
    sA, arrA = cycle_stats(dcA)
    sB, arrB = cycle_stats(dcB)
    sC, arrC = cycle_stats(dcC)
    He_in_shot = src_d.he_per_shot_implanted()
    out["cycle_A_L1_1450"], out["cycle_B_L1_400state"], out["cycle_C_L2_400"] = sA, sB, sC
    out["He_implanted_per_shot"] = He_in_shot
    x = dcA["x"]
    F0 = dcA["fields"][0]
    r_end = np.where(F0["Rvac"] > 0, F0["Rself"] / np.where(F0["Rvac"] > 0, F0["Rvac"], 1.0), np.nan)
    relevant = F0["Rvac"] >= 1e-3 * F0["Rvac"].max()
    out["end_of_cycle_state_1450"] = dict(
        He_mobile_max_appm=float(pr["He"].max()), depth_of_He_max_um=float(f1.x[np.argmax(pr["He"])] * 1e6),
        V_free_max_appm=float(pr["V"].max()), depth_of_V_free_max_um=float(f1.x[np.argmax(pr["V"])] * 1e6),
        He_mobile_total=float((pr["He"] * dcA["w"]).sum()), ratio_depth_integrated=sA["ratio_int_start"],
        ratio_max_over_depth_relevant_cells=float(np.nanmax(np.where(relevant, r_end, np.nan))),
        k11_over_ak=float(4 * C.R_HE1 / C.R_C))
    e1 = out["end_of_cycle_state_1450"]
    log(f"   end-of-cycle state (final Demo baseline, shot {out['checkpoint_shots'][0]}): mobile He max {e1['He_mobile_max_appm']:.2e} appm (x = {e1['depth_of_He_max_um']:.3g} um), "
        f"total {e1['He_mobile_total']:.2e}/m2; free V max {e1['V_free_max_appm']:.2e} appm; self/vacancy trapping rate ratio: depth-integrated {e1['ratio_depth_integrated']:.2e}, "
        f"max over the cells that carry the trapping {e1['ratio_max_over_depth_relevant_cells']:.2e}  (ratio = {e1['k11_over_ak']:.2f} He / sum_k V_k)")
    log(f"   over one cycle from that state: total mobile He peaks at {sA['He_mobile_total_peak']:.2e}/m2 (t = {sA['t_He_mobile_total_peak']:.2e} s; implanted per shot {He_in_shot:.2e}); "
        f"mobile He in a cell up to {sA['He_cell_max_appm']:.1f} appm (x = {sA['depth_of_He_cell_max_um']:.3g} um); ratio depth-integrated max {sA['ratio_int_max']:.2f} at t = {sA['t_ratio_int_max']:.2e} s, "
        f"> 1 between {sA['ratio_int_gt1_t_from']} and {sA['ratio_int_gt1_t_to']} s; max in a cell {sA['ratio_cell_max']:.1f} at x = {sA['depth_of_ratio_cell_max_um']:.3g} um, "
        f"t = {sA['t_of_ratio_cell_max']:.2e} s (He {sA['He_at_ratio_cell_max_appm']:.2e}, sum V_k {sA['Vsum_at_ratio_cell_max_appm']:.2e} appm); "
        f"gross He lost to dimerisation {sA['int_self_trapping_He_per_cycle']:.3e} vs to vacancy trapping {sA['int_vacancy_trapping_He_per_cycle']:.3e} per cycle")
    log(f"   He release in the cycle: thermal detrapping {sA['He_released_by_thermal_detrapping_per_cycle']:.3e}, I+HV annihilation {sA['He_released_by_I_HV_annihilation_per_cycle']:.3e} per m2 "
        f"(T range {sA['T_min']:.0f}-{sA['T_max']:.0f} K)")
    # --- Level-2 channel accounting in the L2 400-shot cycle ----------------------------------------------------------------------
    t2, dx = dcC["t"], dcC["dx"]
    G = {g: np.array([f["groups"][g] for f in dcC["fields"]]) for g in GROUPS + ("breakup_C2", "He_to_HV")}
    ev = {g: tint(t2, G[g], dx) for g in G}
    Hcl = dcC["He_clusters"]
    kpk = int(np.argmax(Hcl))
    kmob = int(np.argmax(dcC["He_mobile"]))
    ch = dict(events_per_shot=ev, He_in_per_shot=float(dcC["He_in"][-1] - dcC["He_in"][0]), Frenkel_vacancies_per_shot=float(dcC["V_prod"][-1] - dcC["V_prod"][0]),
              clusters_start=float(Hcl[0]), clusters_peak=float(Hcl[kpk]), t_clusters_peak=float(t2[kpk]), mobile_He_at_clusters_peak=float(dcC["He_mobile"][kpk]),
              clusters_at_mobile_peak=float(Hcl[kmob]), mobile_peak=float(dcC["He_mobile"][kmob]), clusters_end=float(Hcl[-1]),
              dimer_reversal_fraction=ev["breakup_C2"] / ev["dimerisation"],
              He_via_irreversible_channel_per_shot=ev["He_to_HV"], mutation_events_over_Frenkel=ev["mutation"] / (dcC["V_prod"][-1] - dcC["V_prod"][0]))
    ch["growth_over_dimerisation"] = ev["growth"] / ev["dimerisation"]
    ch["irreversible_over_vacancy_trapping_flux"] = ev["He_to_HV"] / sC["int_vacancy_trapping_He_per_cycle"]
    ch["irreversible_over_implanted"] = ev["He_to_HV"] / ch["He_in_per_shot"]
    out["L2_channel_400"] = ch
    log(f"   L2 cycle (400-shot state): clusters {ch['clusters_start']:.2e} at start, peak {ch['clusters_peak']:.2e} at t = {ch['t_clusters_peak']:.2e} s (mobile He then {ch['mobile_He_at_clusters_peak']:.2e}), "
        f"{ch['clusters_at_mobile_peak']:.2e} when mobile He peaks ({ch['mobile_peak']:.2e}), {ch['clusters_end']:.2e} at the end")
    log("   events per cycle (per m2): " + ", ".join(f"{g} {v:.3e}" for g, v in ev.items()) + f"; dimers re-dissociated: {100 * ch['dimer_reversal_fraction']:.2f} %; "
        f"He through the irreversible channel (vacancy capture + mutation) {ch['He_via_irreversible_channel_per_shot']:.2e} = {ch['irreversible_over_implanted']:.2f} of the implanted He, "
        f"{ch['irreversible_over_vacancy_trapping_flux']:.1e} of the vacancy-trapping flux")
    # --- same cycle, same state, L1 vs L2: totals and where the He ends up ----------------------------------------------------
    def depth_total(dc, f):
        u = dc["y_end"][kn.H:].reshape(f.n, f.S)
        tot = u[:, 0] + (u[:, 3:3 + f.K] * np.arange(1, f.K + 1)[None]).sum(1)
        if f.J:
            tot = tot + (u[:, 3 + f.K:3 + f.K + f.J] * np.arange(2, f.J + 2)[None]).sum(1)
        return tot
    y0u = to_L1(z2["y"], f1, f2)[kn.H:].reshape(f1.n, f1.S)
    H0 = y0u[:, 0] + (y0u[:, 3:3 + f1.K] * np.arange(1, f1.K + 1)[None]).sum(1)
    xx = f1.x * 1e6
    bins = [(0, 0.1), (0.1, 0.5), (0.5, 2), (2, 6), (6, 12), (12, 30), (30, 1000)]
    dep = []
    for lo, hi in bins:
        m = (xx >= lo) & (xx < hi)
        d1 = float(((depth_total(dcB, f1) - H0)[m] * dcB["w"][m]).sum())
        d2 = float(((depth_total(dcC, f2) - H0)[m] * dcC["w"][m]).sum())
        dep.append(dict(bin_um=[lo, hi], dHe_L1=d1, dHe_L2=d2, diff=d2 - d1, diff_over_L1=(d2 - d1) / d1 if d1 else None))
    out["depth_bins_same_state"] = dep
    hp = src_d.he_profile() * src_d.dx                    # He/m2 per shot per cell (source, exact)
    Rv_B = arrB["Rv"]
    rel_B = Rv_B >= 1e-3 * Rv_B.max(1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cellB = np.where(rel_B, np.nan_to_num(arrB["cell"]), 0.0)
    tab = []
    for (lo, hi), dd in zip(bins, dep):
        m = (xx >= lo) & (xx < hi)
        tab.append(dict(bin_um=[lo, hi], He_implanted_share=float(hp[m].sum() / hp.sum()), L1_max_mobile_He_appm=float(arrB["He"][:, m].max()),
                        L1_max_rate_ratio=float(cellB[:, m].max()), L2_peak_cluster_He=float((dcC["He_cl_cell"][:, m] * dcC["w"][m]).sum(1).max()),
                        L1_dHe=dd["dHe_L1"], L2_minus_L1=dd["diff"]))
    out["depth_bin_table"] = tab
    log("   per depth bin (same 400-shot state): " + "; ".join(f"{t_['bin_um'][0]}-{t_['bin_um'][1]} um: implanted {100 * t_['He_implanted_share']:.1f} %, max mobile He {t_['L1_max_mobile_He_appm']:.1e} appm, "
        f"max rate ratio {t_['L1_max_rate_ratio']:.1e}, peak cluster He {t_['L2_peak_cluster_He']:.1e}" for t_ in tab))
    out["max_depth_bin_relative_difference"] = float(max(abs(d["diff_over_L1"]) for d in dep if d["diff_over_L1"] is not None))
    out["max_mobile_He_in_cycle_L1_vs_L2"] = [float(dcB["He_mobile"].max()), float(dcC["He_mobile"].max())]
    log("   same state, one cycle, change of total He per depth bin (He/m2), L1 / L2 / (L2-L1)/L1: " +
        "; ".join(f"{d['bin_um'][0]}-{d['bin_um'][1]} um {d['dHe_L1']:.3e} {d['dHe_L2']:.3e} {100 * (d['diff_over_L1'] or 0):+.2f} %" for d in dep))
    # --- is the residual difference numerical?  one shot from the same state at several tolerances -----------------------------------
    log("   one shot from the SAME state (400-shot L2 state, clusters dropped for L1) at several tolerances:")
    out["same_state_tolerance_test"] = one_shot_same_state(src_d, z2["y"], f1, f2, ((1e-3, 1e-9),) if quick else ((1e-3, 1e-9), (1e-4, 1e-9), (1e-5, 1e-10)))
    # --- fresh paired runs at the production tolerance: reproducibility of the stored totals -------------------------------------------
    tol = {}
    res = {}
    for lab, p in (("L1", L1P), ("L2", L2P)):
        sim = FastSimulation("Demo", p, CYC, sources=src_d)
        lst = []
        for _ in range(2):
            sim.shot()
            lst.append(sim.snapshot())
        res[lab] = lst
        exact_per_shot = src_d.he_per_shot_implanted() + float((sim.fm.bgHe * sim.P * sim.fm.dx).sum() * 1e-6 * NC)
    tol = dict(rel_L2_L1=[res["L2"][k]["He_retained"] / res["L1"][k]["He_retained"] - 1 for k in range(2)],
               He_in_error_L1=[res["L1"][k]["He_in"] / (exact_per_shot * (k + 1)) - 1 for k in range(2)],
               He_in_error_L2=[res["L2"][k]["He_in"] / (exact_per_shot * (k + 1)) - 1 for k in range(2)],
               L1_vs_stored=[float(res["L1"][k]["He_retained"] / a["He_retained"][k] - 1) for k in range(2)],
               L2_vs_stored=[float(res["L2"][k]["He_retained"] / b["He_retained"][k] - 1) for k in range(2)])
    out["fresh_pair_rtol_1e-3_2_shots"] = tol
    log(f"   fresh 2-shot pair at the production tolerance: L2/L1-1 = {tol['rel_L2_L1']}; integrated source (known exactly) off by {tol['He_in_error_L1']} (L1), {tol['He_in_error_L2']} (L2); "
        f"fresh vs stored retained He: L1 {tol['L1_vs_stored']}, L2 {tol['L2_vs_stored']}")
    cv = json.loads((VER / "results.json").read_text(encoding="utf-8")).get("test5_convergence", {}).get("details", {})
    out["convergence_test5_30_shots"] = {k: v.get("rel_change_retained") for k, v in cv.items() if isinstance(v, dict) and "rel_change_retained" in v}
    log("   stored convergence test (30 shots, change of retained He):", out["convergence_test5_30_shots"])
    # arrays for the figure
    ok = arrA["He"].max(0) > 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cell_max_t = np.nanmax(arrA["cell"], 0)
    out["profiles"] = dict(x_um=x, He_end=pr["He"], V_end=pr["V"], ratio_end=r_end, He_cycle_max=arrA["He"].max(0), V_cycle_max=arrA["Vf"].max(0),
                           ratio_cycle_max=np.where(ok, cell_max_t, np.nan),
                           t=dcB["t"], mobile_L1=dcB["He_mobile"], mobile_L2=dcC["He_mobile"], clusters_L2=dcC["He_clusters"], ratio_int_L1=arrB["ratio_int"])
    return out


# ======================================================================================================
# figure
# ======================================================================================================
def figure(R1, R2, R3, path=None):
    plt = style()
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    c1, c2 = "#2a6fb0", "#c8402b"
    # (a) forced-on over time
    a = ax[0, 0]
    tt = {k: np.array([r["t"] for r in v]) for k, v in R2["time"].items()}
    for k, col, ls in (("L1", c1, "--"), ("L2", c2, "-")):
        a.loglog(tt[k], [max(r["He_retained"], 1.0) for r in R2["time"][k]], color=col, ls=ls, lw=2, label=f"{k}: retained He")
    a.loglog(tt["L2"], [max(r["He_clusters"], 1.0) for r in R2["time"]["L2"]], color=c2, ls=":", lw=2, label="L2: He in He$_{2..7}$ clusters")
    a.loglog(tt["L2"], [max(r["V_net"], 1.0) for r in R2["time"]["L2"]], color="#7b3fa0", ls="-.", lw=2, label="L2: vacancies made by trap mutation")
    a.set(xlabel="time in the cycle (s)", ylabel="per m$^2$", ylim=(1e5, 3e19), xlim=(1e-7, 1.0),
          title="(a) forced on, shot 1: 1000 K, $\\xi$ = 0, he_scale = 100")
    a.legend(fontsize=9, loc="lower right")
    # (b) baseline inside one cycle, same start state
    b = ax[0, 1]
    P = R3["profiles"]
    t = P["t"][1:]
    b.loglog(t, np.maximum(P["mobile_L1"][1:], 1.0), color=c1, ls="--", lw=2, label="L1 mobile He")
    b.loglog(t, np.maximum(P["mobile_L2"][1:], 1.0), color=c2, lw=2, label="L2 mobile He")
    b.loglog(t, np.maximum(P["clusters_L2"][1:], 1.0), color=c2, ls=":", lw=2, label="L2 He in clusters")
    b.set(xlabel="time in the Demo cycle (s)", ylabel="He per m$^2$", xlim=(1e-7, 0.1), ylim=(1e6, 1e20),
          title="(b) Demo baseline state (shot 400), one cycle")
    b2 = b.twinx()
    b2.semilogx(t, P["ratio_int_L1"][1:], color="0.35", lw=1.5, label="self / vacancy trapping (L1 state, right)")
    b2.axhline(1.0, color="0.35", lw=0.8, ls=":")
    b2.set(ylabel="rate ratio, integrated over depth", ylim=(0, 8))
    b2.grid(False)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2_ = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2_, fontsize=8.5, loc="lower left")
    # (c) depth profiles (baseline final state)
    c = ax[0, 2]
    x = P["x_um"]
    c.loglog(x, np.maximum(P["He_end"], 1e-30), color=c1, lw=2, label="mobile He, end of cycle")
    c.loglog(x, np.maximum(P["He_cycle_max"], 1e-30), color=c1, ls="--", lw=2, label="mobile He, max over the cycle")
    c.loglog(x, np.maximum(P["V_end"], 1e-30), color="k", lw=2, label="free V, end of cycle")
    c.loglog(x, np.maximum(P["V_cycle_max"], 1e-30), color="k", ls="--", lw=2, label="free V, max over the cycle")
    c.set(xlabel="depth ($\\mu$m)", ylabel="concentration (appm)", xlim=(1e-3, 60), ylim=(1e-12, 1e3),
          title="(c) final Demo baseline state (1450 shots)")
    c.legend(fontsize=9, loc="lower left")
    # (d) rate ratio vs depth
    d = ax[1, 0]
    d.loglog(x, P["ratio_end"], color=c1, lw=2, label="end of cycle (stored final state)")
    d.loglog(x, P["ratio_cycle_max"], color=c1, ls="--", lw=2, label="max over the cycle (450 sampled times)")
    d.axhline(1.0, color="k", lw=1)
    d.text(1.2e-3, 1.5, "self-trapping = vacancy trapping", fontsize=9)
    d.set(xlabel="depth ($\\mu$m)", ylabel="He lost to self-trapping / to vacancy trapping", xlim=(1e-3, 60), ylim=(1e-12, 1e3),
          title="(d) rate ratio $k_{11}$He$^2$ / $\\sum_k a_k$He V$_k$ vs depth")
    d.legend(fontsize=9, loc="upper right")
    # (e) Demo he_scale scan
    e = ax[1, 1]
    sc = R2["demo_scan"]
    hs = np.array([r["he_scale"] for r in sc])
    rel = np.array([r["rel_diff"] for r in sc]) * 100
    e.loglog(hs, np.maximum(np.abs(rel), 1e-5), "o-", color=c2, lw=2, ms=8, label="Demo, 2 shots from empty: |L2/L1 - 1|")
    e.loglog([1.0], [abs(R3["stored_runs"][400]["rel"]) * 100], "D", mfc="none", color="k", ms=10, label="stored 400-shot runs (he_scale 1)")
    rt = R3["same_state_tolerance_test"]
    tight = rt[sorted(rt, key=float)[0]]
    e.loglog([1.0], [abs(rt[sorted(rt, key=float)[-1]]["rel_increment"]) * 100], "s", color="#2d8a4e", ms=9, label="one shot, same state, tightest rtol")
    e.set(xlabel="he_scale (He source multiplier)", ylabel="relative difference of retained He (%)", ylim=(1e-4, 1e4),
          title="(e) Level-2 effect vs He/vacancy ratio")
    e.legend(fontsize=9, loc="upper left")
    # (f) term check
    f = ax[1, 2]
    cs = R1["cases"]
    vals = [max(x_["rel_err"], 1e-17) for x_ in cs]
    yy = np.arange(len(cs))[::-1]
    f.barh(yy, vals, color="#4c9a5b")
    f.set_yticks(yy)
    f.set_yticklabels([f"{x_['short'][:34]} [{'T-profile' if 'T=700' in x_['case'] else '1000K'}]" for x_ in cs], fontsize=8)
    f.set_xscale("log")
    f.set(xlim=(1e-17, 1e-8), xlabel="max |kernel $-$ analytic| / max |analytic|",
          title="(f) each Level-2 term vs its formula")
    f.axvline(1e-9, color="k", ls="--")
    f.text(1.3e-9, len(cs) - 0.6, "threshold", fontsize=9)
    fig.suptitle("Test 7: Level-2 switch (He self-trapping + trap mutation): forced on, and what it does in the Demo baseline", fontsize=15)
    fig.tight_layout()
    fig.savefig(str(path or (VER / "test7_level2.pdf")))
    import tempfile
    fig.savefig(str(Path(tempfile.gettempdir()) / "test7_level2_preview.png"), dpi=60)    # preview outside the repository
    log("   figure saved")


def build_record(R1, R2, R3):
    forced_ok = bool(R2["headline_rel_diff"] > 0.05)
    max_cons = max(R2["max_cons"], R3["L2_run_cons_max"], R3["L1_run_cons_max"])
    cons_ok = bool(max_cons < 1e-10)
    passed = forced_ok and cons_ok and R1["passed"]
    h = R2["headline"]
    r400 = R3["stored_runs"][400]
    st = R3["same_state_tolerance_test"]
    kw = dict(forced_on_rel_diff_retained_He=R2["headline_rel_diff"], forced_on_threshold=0.05,
              forced_on_L1_retained=h["L1"][0]["He_retained"], forced_on_L2_retained=h["L2"][0]["He_retained"],
              forced_on_L2_He_clusters=h["L2"][0]["He_clusters"], forced_on_L2_vacancies_from_mutation=h["L2"][0]["V_net"],
              max_He_conservation_error=max_cons, term_check_max_rel_err=R1["max_rel_err_terms"], term_check_passed=R1["passed"],
              baseline_L2_vs_L1_400_shots=r400["rel"],
              baseline_rate_ratio_end_of_cycle_depth_integrated=R3["end_of_cycle_state_1450"]["ratio_depth_integrated"],
              baseline_rate_ratio_max_in_cycle_depth_integrated=R3["cycle_A_L1_1450"]["ratio_int_max"],
              baseline_same_state_one_shot_increment_rel_diff={k: v["rel_increment"] for k, v in st.items()},
              details=tolist(dict(unit=R1, forced=dict((k, v) for k, v in R2.items() if k != "time"),
                                  baseline=dict((k, v) for k, v in R3.items() if k != "profiles"))),
              note="forced on: Prototype spectrum, constant T, xi = 0, no neutrons, P = 1 s; headline 1000 K, he_scale 100")
    return passed, kw


def main():
    args = set(sys.argv[1:])
    quick = "quick" in args
    which = {a for a in args if a in ("p1", "p2", "p3", "fig")} or {"p1", "p2", "p3", "fig"}
    everything = which == {"p1", "p2", "p3", "fig"}
    R1 = R2 = R3 = None
    src_d = src_p = None
    if "p1" in which:
        R1 = part1()
    if which & {"p2", "p3"}:
        e = make_grid()
        if "p2" in which:
            src_p = ShotSources("Prototype", e)
        src_d = ShotSources("Demo", e)
        log("sources built")
    if "p2" in which:
        R2 = part2(src_p, src_d, quick)
    if "p3" in which:
        R3 = part3(src_d, quick)
    if everything and not quick:
        figure(R1, R2, R3)
        passed, kw = build_record(R1, R2, R3)
        record("test7_level2", passed, **kw)
    elif everything and quick:           # dry run of the figure and of the record (nothing written into the repository)
        import tempfile
        figure(R1, R2, R3, path=Path(tempfile.gettempdir()) / "test7_quick.pdf")
        passed, kw = build_record(R1, R2, R3)
        json.dumps(kw, default=float)
        log(f"quick dry run complete (passed would be {passed}); nothing recorded")
    else:
        log("partial run: nothing recorded")


if __name__ == "__main__":
    main()
