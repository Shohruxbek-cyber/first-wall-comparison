"""Rounding of short time steps at large clock times (revision of 21 Oct 2026, Appendix C).

The solver advances t1 = t0 + dt and spreads the source energy of the step over t1 - t0, while the
cell update uses the requested dt; in double precision the two differ when dt is small against t0.
This script tabulates (t1 - t0)/dt - 1 for the 0.5 ns steps that resolve the 5 ns neutron pulse (the
step limit of a square pulse is 0.1 of its duration) and the 2 ns steps of the 20 ns x-ray pulse, at
the arrival times of the pulses in every shot of the repeated tungsten runs (Demo up to 1090 shots at
10 Hz, Prototype 100 shots at 1 Hz), and the mean over the shots, which is what a run accumulates.
The replay of the neutron pulses covers the neutron-heated shots of each run with neutron heating
(results/neutron/w_multi_<scenario>_steel<f>/: Demo resumed after shot 600, until the shot at which
the run stopped) and is set against the energy residual of that run's ledger (its final checkpoint).
It runs no solver.  Usage: python scripts/diag_clock_rounding.py [--out=results/revision/clock_rounding]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.neutron import neutron_arrival  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

C = 2.99792458e8


def pulse_error(t_start, P_shift, duration, fluence, dt0=1e-13, dt_rel=0.02, dt_cap=np.inf):
    """Replay the step sequence of ThermalSolver.run over one square pulse that starts at the absolute
    clock time t_start (an event: the step is reset to dt0, then limited to dt_rel of the time since
    the event and to dt_cap) and return the energy applied to the cells minus the energy booked,
    with the energy of a step computed as by the Periodic source: from the clock readings shifted by
    the shot start P_shift, spread over t1 - t0 and applied over the requested dt."""
    t, t_event, dt, err, end = float(t_start), float(t_start), dt0, 0.0, float(t_start) + duration
    s0 = float(t_start) - P_shift  # pulse start in shifted time, as the source sees it
    while t < end:
        lim = min(dt_cap, max(dt_rel * (t - t_event), dt0))
        dt = min(dt * 1.3, lim, end - t) if t > t_event else dt0
        t1 = t + dt
        a, b = max(t - P_shift, s0), min(t1 - P_shift, s0 + duration)
        e = fluence * max(b - a, 0.0) / duration
        if t1 > t:
            err += e * dt / (t1 - t) - e
        t = t1
    return err


def rel_errors(t_starts, dt, n_steps):
    """(t1 - t0)/dt - 1 for n_steps consecutive steps of length dt from each start time."""
    out = []
    for t0 in t_starts:
        t = float(t0)
        for _ in range(n_steps):
            t1 = t + dt
            out.append((t1 - t) / dt - 1.0)
            t = t1
    return np.array(out)


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "clock_rounding"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, n_shots, first in (("Demo", 1090, 600), ("Prototype", 100, 0)):
        sc = SCENARIOS[name]
        P = 1.0 / sc.rep_Hz
        k = np.arange(first, n_shots)
        t_n = k * P + neutron_arrival(sc.R)
        t_x = k * P + sc.R / C
        e_n = rel_errors(t_n, 0.1 * 5e-9, 10)
        e_x = rel_errors(t_x, 0.1 * 20e-9, 10)
        res[name] = {"shots_with_neutron_heating": [int(first) + 1, int(n_shots)],
                     "clock_at_last_neutron_pulse_s": float(t_n[-1]),
                     "neutron_steps_0.5ns": {"max_abs": float(np.abs(e_n).max()), "mean": float(e_n.mean())},
                     "xray_steps_2ns": {"max_abs": float(np.abs(e_x).max()), "mean": float(e_x.mean())}}
        # replay of the steps over each neutron pulse (energy per shot from the source definition)
        from ife_wall.neutron import E_W_REF, R_REF, YIELD_REF
        e_W = E_W_REF * sc.yield_J / YIELD_REF * (R_REF / sc.R) ** 2
        for f in (0.5, 1.0):
            fluence = e_W * 1e-3 + f * e_W * 1e-2  # 1 mm W + 1 cm steel, J/m^2 per shot
            run = ROOT / "results" / "neutron" / f"w_multi_{name}_steel{f:g}"
            n_run = int(json.loads((run / f"{name}.json").read_text(encoding="utf-8"))["n_shots"])
            kr = np.arange(first, n_run)  # the shots of this run with neutron heating
            err = sum(pulse_error(kk * P + neutron_arrival(sc.R), kk * P, 5e-9, fluence) for kk in kr)
            led = json.loads(str(np.load(run / f"{name}_checkpoint.npz")["ledger"]))  # final state of the run
            resid = (led["deposited"] + led["front_flux_in"] - led["to_coolant"] - led["vaporized_latent"]
                     - led["vaporized_sensible"] - (led["stored_final"] - led["stored_initial"]))  # EnergyLedger.residual
            res[name][f"replayed_neutron_error_steel{f:g}"] = {
                "shots_with_neutron_heating": [int(first) + 1, n_run], "neutron_J_m2_per_shot": fluence,
                "error_J_m2_total": err, "error_per_shot_rel_to_neutron_energy": err / len(kr) / fluence,
                "run_residual_J_m2": resid, "run_residual_per_shot_rel_to_neutron_energy": resid / len(kr) / fluence,
                "share_of_run_residual_explained": -err / resid}
        print(name, json.dumps(res[name]))
    (out / "clock_rounding.json").write_text(json.dumps({
        "note": "(t1 - t0)/dt - 1 in double precision for 10 consecutive steps per pulse and shot; the solver spreads a step's "
                "source energy over t1 - t0 and applies it over dt, so the deposited energy of such a step differs from the "
                "booked one by this relative amount", "results": res}, indent=1), encoding="utf-8")
