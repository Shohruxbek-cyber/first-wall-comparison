"""Record T(depth, t) in the 1 mm tungsten armour for one shot cycle, for the He-retention model.

Re-runs the EXISTING thermal solver (ife_wall.scenarios.w_case, unchanged) for one shot starting
from a uniform pre-shot temperature, and stores the temperature at a fixed set of depths after
every accepted time step (50 us at fine resolution, then up to the next shot).

Usage: python scripts/record_he_thermal.py <Experimental|Prototype|Demo> <T_init_K|cold> [--P=seconds]
  T_init_K  uniform pre-shot temperature (the converged pre-shot surface temperature gives the
            quasi-steady cycle); 'cold' uses the scenario's own initial temperature (true first shot);
            'converged' the converged pre-shot surface temperature of the scenario's repeated-shot run
            (ife_wall.scenarios.converged_preshot; not for the spectra used before 22 Oct 2026)
  --Tback=  back-face steel temperature of the quasi-steady start (steel linear from T_init to this value AT the
            back face); --Tback=converged: the true back-face temperature of the same converged state
            (ife_wall.scenarios.converged_back_face).  Before 22 Oct 2026 the value given here was the centre of
            the last steel cell of the per-shot record (701.2 K Demo), not the face
  --P=      cycle length; default 1/rep rate.  Experimental has no rep rate: 1 s is an ASSUMPTION.
Output: runs/he_retention/thermal/<scenario>_T<T_init>.npz  (t, z, T[float32 nt x nz], meta)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, Options, converged_back_face, converged_preshot, w_case  # noqa: E402

# depth grid for the stored temperatures [m]: surface, then 1 nm ... 1 mm (log spaced)
Z = np.r_[0.0, np.geomspace(1e-9, 1e-3, 240)]


def record_cycle(name, T_init, P, T_back=None):
    scen = SCENARIOS[name]
    opt = Options(T_init=None if T_init is None else float(T_init))
    s = w_case(scen, opt)
    if T_back is not None:
        # quasi-steady start: W at T_init, steel linear from T_init (interface) to T_back (back face)
        T0 = np.full(len(s.m), float(T_init))
        st = np.nonzero(s.layer == s.layer.max())[0]
        x = np.cumsum(s.m[st]) - s.m[st] / 2
        T0[st] = T_init + (T_back - T_init) * x / (x[-1] + s.m[st][-1] / 2)
        s.H = s._by_layer(lambda mat, sl: mat.h_of_T(T0[sl]))
        s.ledger.stored_initial = s.stored_energy()
    ts, Ts = [], []

    def sample(sv):
        T = sv.T
        xs = np.r_[0.0, sv.x_centers()]
        Tsurf = np.r_[sv.surface_temperature(T), T]
        ts.append(sv.t)
        Ts.append(np.interp(Z, xs, Tsurf).astype(np.float32))

    sample(s)
    t0 = time.time()
    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=sample)
    s.run(P, dt0=1e-9, dT_target=5.0, record=sample)
    meta = dict(scenario=name, T_init=float(s.T[0] if T_init is None else T_init), P=P, n_steps=len(ts),
                T_surface_start=float(Ts[0][0]), T_surface_end=float(Ts[-1][0]),
                T_5um_start=float(np.interp(5e-6, Z, Ts[0])), T_5um_end=float(np.interp(5e-6, Z, Ts[-1])),
                T_surface_peak=float(max(a[0] for a in Ts)), wall_s=time.time() - t0,
                energy_rel_residual=float(s.ledger.residual()[1]), mass_evaporated_kg_m2=float(s.mass_evaporated))
    return np.array(ts), np.array(Ts), meta


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, tinit = args[0], args[1]
    Pa = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("--P=")]
    scen = SCENARIOS[name]
    P = float(Pa[0]) if Pa else (1.0 / scen.rep_Hz if scen.rep_Hz else 1.0)
    Ti = None if tinit == "cold" else converged_preshot(name) if tinit == "converged" else float(tinit)
    Tb = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("--Tback=")]
    Tb = [str(converged_back_face(name)) if b == "converged" else b for b in Tb]
    t, T, meta = record_cycle(name, Ti, P, float(Tb[0]) if Tb else None)
    meta["T_back_start"] = float(Tb[0]) if Tb else None
    d = ROOT / "runs" / "he_retention" / "thermal"
    d.mkdir(parents=True, exist_ok=True)
    tag = "cold" if Ti is None else f"{Ti:.1f}"
    np.savez_compressed(d / f"{name}_T{tag}{'_ss' if Tb else ''}.npz", t=t, z=Z, T=T, meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)
