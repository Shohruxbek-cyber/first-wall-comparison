"""Demo with the pre-shot temperature rising from 600 K (true first shots), as in the existing 600-shot thermal run.

The recorded quasi-steady cycle (pre-shot T_SS, ife_wall.scenarios.converged_preshot) is shifted for shot k by
T_pre(k) - T_SS,  where T_pre(k) is
the pre-shot surface temperature of the existing multi-shot run (runs/w_multi/baseline/Demo_per_shot.csv).  The peak
surface temperature of that run rises by about as much as the pre-shot temperature (285 against 321 K with the
spectra used before 22 Oct 2026), so a rigid shift is a good approximation (error < 40 K in the spike then).  Output: runs/he_retention/sens/Demo_T_rising/.
Usage: python scripts/run_he_rising_T.py [n_shots]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall.he import cases as CS  # noqa: E402
from ife_wall.he.simulate import FastSimulation  # noqa: E402
from ife_wall.scenarios import converged_preshot  # noqa: E402
from run_he_retention import COLS  # noqa: E402

n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 300
per = np.genfromtxt(ROOT / "runs/w_multi/baseline/Demo_per_shot.csv", delimiter=",", names=True)
Tpre = per["preshot_surface_T_K"]
params, kw = CS.build("baseline")
T_SS = converged_preshot("Demo", legacy=921.1)  # pre-shot temperature of the recorded quasi-steady cycle
sim = FastSimulation("Demo", params, str(ROOT / f"runs/he_retention/thermal/Demo_T{T_SS:.1f}_ss.npz"), **kw)
Tc0 = sim.fm.Tc.copy()
out = ROOT / "runs/he_retention/sens/Demo_T_rising"
out.mkdir(parents=True, exist_ok=True)
rows = []
t0 = time.time()
for k in range(n_shots):
    off = float(Tpre[min(k, len(Tpre) - 1)] - T_SS)
    sim.fm.set_cycle(sim.fm.tT, Tc0 + off)
    sim.shot()
    sn = sim.snapshot()
    rows.append([sn[c] for c in COLS])
    if (k + 1) % 25 == 0:
        np.savetxt(out / "totals.csv", np.array(rows), delimiter=",", header=",".join(COLS), comments="")
        print(f"shot {k + 1}: offset {off:.0f} K retained {sn['He_retained']:.4e} ({time.time() - t0:.0f}s)", flush=True)
np.savetxt(out / "totals.csv", np.array(rows), delimiter=",", header=",".join(COLS), comments="")
(out / "meta.json").write_text('{"case": "T_rising", "note": "pre-shot T of the existing 600-shot run, rigid shift of the quasi-steady cycle"}', encoding="utf-8")
