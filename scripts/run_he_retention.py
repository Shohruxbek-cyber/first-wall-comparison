"""Run the He-retention model for one scenario and one case (reproducible entry point).

  python scripts/run_he_retention.py Demo --shots 3000 --case baseline
  python scripts/run_he_retention.py Prototype --shots 1000 --case Eb_low --out runs/he_retention/Prototype_Eb_low
  python scripts/run_he_retention.py Demo --shots 200 --case baseline --set rtol=1e-4 --grid fine

Scenarios: Experimental (single-shot design, adiabatic, no repetition rate: use --shots 1; multi-shot Experimental runs were an assumption and are not reported),
Prototype (1 Hz), Demo (10 Hz).  Temperature: the recorded quasi-steady cycle from
scripts/record_he_thermal.py (run it first; the files are kept in runs/he_retention/thermal).
Output (default runs/he_retention/<scenario>_<case>/): totals.csv (every shot), profiles.npz,
checkpoint.npz (resume with --resume), meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.he import cases as CS  # noqa: E402
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.simulate import FastSimulation  # noqa: E402
from ife_wall.scenarios import converged_preshot  # noqa: E402

CYCLE = {"Demo": lambda: f"Demo_T{converged_preshot('Demo', legacy=921.1):.1f}_ss",  # evaluated when used
         "Prototype": lambda: f"Prototype_T{converged_preshot('Prototype', legacy=609.3):.1f}_ss",
         "Experimental": lambda: "Experimental_Tcold"}
COLS = ["shot", "He_retained", "He_mobile", "He_trapped", "He_clusters", "He_surface", "He_in", "He_cons_err",
        "V_free", "V_He", "I", "V_surface", "V_prod"]


def grid(kind):
    if kind == "production":
        return make_grid()
    if kind == "fine":   # every production cell split in two (convergence test)
        from ife_wall.he.grid import refine
        return refine(make_grid(), 2)
    raise ValueError(kind)


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(CYCLE))
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--case", default="baseline", choices=list(CS.CASES))
    ap.add_argument("--cycle", default=None, help="thermal npz (default: quasi-steady cycle of the scenario)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", default="production", choices=["production", "fine"])
    ap.add_argument("--profiles", default="1,10,100,300,1000,3000,10000")
    ap.add_argument("--set", nargs="*", default=[], help="FastSimulation overrides, e.g. rtol=1e-4 xi=0.5")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds (state is checkpointed)")
    a = ap.parse_args()
    extra = {k: parse_val(v) for k, v in (s.split("=") for s in a.set)}
    params, sim_kw = CS.build(a.case, **extra)
    cyc = a.cycle or str(ROOT / "runs" / "he_retention" / "thermal" / (CYCLE[a.scenario]() + ".npz"))
    out = Path(a.out) if a.out else ROOT / "runs" / "he_retention" / f"{a.scenario}_{a.case}"
    out.mkdir(parents=True, exist_ok=True)
    sim = FastSimulation(a.scenario, params, cyc, edges=grid(a.grid), **sim_kw)
    ck = out / "checkpoint.npz"
    rows, profs = [], {}
    if a.resume and ck.exists():
        z = np.load(ck, allow_pickle=True)
        sim.y, sim.shot_no = z["y"], int(z["shot_no"])
        rows = [list(r) for r in z["rows"]]
        profs = z["profs"].item()
        print(f"resumed at shot {sim.shot_no}", flush=True)
    want_prof = {int(s) for s in a.profiles.split(",") if s}
    t0 = time.time()
    while sim.shot_no < a.shots:
        sim.shot()
        k = sim.shot_no
        sn = sim.snapshot()
        rows.append([sn[c] for c in COLS])
        if k in want_prof or k == a.shots:
            profs[k] = sim.fm.profiles(sim.y)
        stop = bool(a.time_limit and time.time() - t0 > a.time_limit)
        if k % 25 == 0 or k == a.shots or stop:
            np.savetxt(out / "totals.csv", np.array(rows), delimiter=",", header=",".join(COLS), comments="")
            np.savez(ck, y=sim.y, shot_no=sim.shot_no, rows=np.array(rows), profs=np.array(profs, dtype=object))
            print(f"[{a.scenario}/{a.case}] shot {k}: retained {sn['He_retained']:.4e}/m2 "
                  f"({100 * sn['He_retained'] / sn['He_in']:.1f} % of implanted), cons err {sn['He_cons_err']:.1e}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if stop:
            break
    np.savetxt(out / "totals.csv", np.array(rows), delimiter=",", header=",".join(COLS), comments="")
    last = {c: float(v) for c, v in zip(COLS, rows[-1])}
    meta = dict(scenario=a.scenario, case=a.case, shots_done=sim.shot_no, grid=a.grid, cycle=cyc, last=last,
                params={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in vars(params).items() if k != "extra"},
                sim=sim_kw, wall_s=time.time() - t0, P=sim.P, neutron_scale=sim.neutron_scale,
                source_summary=dict(He_incident_per_shot=sim.src.he_per_shot_incident(),
                                    He_implanted_per_shot=sim.src.he_per_shot_implanted(),
                                    displacements_per_m2_per_shot=sim.src.displacements_per_m2()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    np.savez(out / "profiles.npz", **{f"{s}_{k}": v for s, p in profs.items() for k, v in p.items()})


if __name__ == "__main__":
    main()
