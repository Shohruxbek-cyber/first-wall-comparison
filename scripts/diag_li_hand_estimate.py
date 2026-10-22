"""Outside check of the lithium evaporation (revision of 20 Oct 2026): a hand estimate
of the peak surface temperature and of the evaporated mass of a thick Demo film from an energy
balance, independent of the solver, to compare with the computed 2224 K and 7.14e-4 kg/m^2.

Assumptions:
  * Only the debris ions heat the surface.  They stop within about a micrometre, less than the
    diffusion length over their arrival window, so their energy E_d is treated as a surface heat
    flux.  Burn ions (MeV, ranges of hundreds of micrometres) and x-rays (9 % absorbed in 1 mm of
    lithium) are ignored.
  * Mean flux q_mean = 0.8 E_d / (t90 - t10) over the 10-90 % debris arrival window; peak flux
    q_peak = the largest debris arrival power.
  * Semi-infinite liquid lithium with the solver's Davison (1968) correlations, evaluated at the
    mean of 600 K and the peak temperature; HKL flux with alpha = 1.
  * Peak surface temperature: (i) without evaporation, T0 + 2 q_mean sqrt(dt / pi) / e with
    e = sqrt(k rho cp); (ii) capped by evaporation where j(T) (L_v + cp (T - T0)) = q, for q_mean
    and q_peak (conduction neglected, so these caps are upper estimates for that flux).
  * Evaporated mass: (E_d - Q_cond) / (L_v + cp (T_cap - T0)), with Q_cond = 2 e (T_cap - T0)
    sqrt(dt / pi) the energy conducted into the liquid if the surface were held at the q_mean cap
    for the whole window (an overestimate, so the mass is a low estimate); and the upper bound
    E_d / (L_v + cp (T_cap - T0)) with no conduction at all.

Usage: python scripts/diag_li_hand_estimate.py [--out=results/revision/li_hand_estimate]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.library import lithium  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

T0 = 600.0


def debris_arrival(scen):
    ions = IonSpectrumSource(load_ion_bins(True), scen.yield_J, scen.R, ["Li", "Fe"], srim.SrimStopping("projected"))
    deb = np.array([b.group == "debris" for b in ions.bins])
    t = np.geomspace(ions.t_first, ions.t_last, 4001)
    E = np.array([float(ions.energy_components(ions.t_first, ti)[deb].sum()) for ti in t])
    P = np.diff(E) / np.diff(t)
    frac = E / E[-1]
    t10, t90 = float(np.interp(0.1, frac, t)), float(np.interp(0.9, frac, t))
    return float(E[-1]), t10, t90, float(P.max()), float(t[1:][np.argmax(P)])


def debris_depths(scen):
    """Depths (um) within which 50 % and 90 % of the debris-ion energy is deposited in a thick
    film at 600 K, and the energy-weighted mean depth, from the production depth model."""
    from ife_wall.scenarios import li_case
    s = li_case(scen, 1e-3, T0)
    ions = s.sources[1]
    fr = np.asarray(ions.fractions(s))
    deb = np.array([b.group == "debris" for b in ions.bins])
    prof = ions.fluence[deb] @ fr[deb]                     # J/m^2 per cell
    x = s.x_centers()
    edges = np.r_[0.0, np.cumsum(s.dx())]
    cum = np.cumsum(prof) / prof.sum()
    return {"mean_um": float((prof * x).sum() / prof.sum() * 1e6),
            "d50_um": float(np.interp(0.5, cum, edges[1:]) * 1e6), "d90_um": float(np.interp(0.9, cum, edges[1:]) * 1e6),
            "fraction_of_debris_energy_in_film": float(prof[s.layer == 0].sum() / ions.fluence[deb].sum())}


def props(Li, T):
    T = np.array([T])
    return float(Li.k(T)[0]), float(Li.rho(T)[0]), float(Li.cp(T)[0])


def cap(Li, q):
    """Surface temperature at which evaporation alone removes the flux q."""
    v = Li.vapor
    cp = props(Li, 1000.0)[2]
    g = lambda T: float(v.hkl_flux(np.array([T]), 1.0)[0]) * (float(v.L_v(np.array([T]))[0]) + cp * (T - T0)) - q  # noqa: E731
    return brentq(g, 1000.0, 4000.0, xtol=1e-6)


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_hand_estimate"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    scen = SCENARIOS["Demo"]
    Li = lithium()
    E_d, t10, t90, P_max, t_Pmax = debris_arrival(scen)
    dt = t90 - t10
    q_mean = 0.8 * E_d / dt
    T_cap_mean, T_cap_peak = cap(Li, q_mean), cap(Li, P_max)
    k, rho, cp = props(Li, 0.5 * (T0 + T_cap_mean))
    e = math.sqrt(k * rho * cp)
    T_noevap = T0 + 2 * q_mean * math.sqrt(dt / math.pi) / e
    L_v = float(Li.vapor.L_v(np.array([T_cap_mean]))[0])
    per_kg = L_v + cp * (T_cap_mean - T0)
    Q_cond = 2 * e * (T_cap_mean - T0) * math.sqrt(dt / math.pi)
    model = json.loads((ROOT / "runs" / "summary" / "summary.json").read_text(encoding="utf-8"))["Li"]["Demo"]
    rows = model["rows"]
    dd = debris_depths(scen)
    res = {"assumptions": __doc__.split("Usage:")[0].strip(),
           "debris_energy_J_m2": E_d, "debris_window_10_90_us": [t10 * 1e6, t90 * 1e6],
           "debris_deposition_depth": dd, "diffusion_length_over_window_um": math.sqrt(k / (rho * cp) * dt) * 1e6,
           "q_mean_W_m2": q_mean, "q_peak_W_m2": P_max, "t_q_peak_us": t_Pmax * 1e6,
           "effusivity_SI": e, "properties_at_K": 0.5 * (T0 + T_cap_mean),
           "T_surface_no_evaporation_K": T_noevap,
           "T_surface_evaporation_cap_mean_flux_K": T_cap_mean, "T_surface_evaporation_cap_peak_flux_K": T_cap_peak,
           "conducted_J_m2_if_held_at_cap": Q_cond,
           "evaporated_kg_m2_estimate": max(E_d - Q_cond, 0.0) / per_kg,
           "evaporated_kg_m2_upper_bound_no_conduction": E_d / per_kg,
           "model_thick_film": {"evaporated_kg_m2": model["vaporized_kg_m2_thick_film"],
                                "peak_surface_K_1mm": rows[-1]["peak_Li_surface_K"]}}
    (out / "li_hand_estimate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "assumptions"}, indent=1))
