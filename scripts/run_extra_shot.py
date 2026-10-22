"""One extra Demo shot from a multi-shot checkpoint, keeping the full time history.

Loads <run>/<scenario>_checkpoint.npz written by run_w_multi.py (read only; the checkpoint is
not modified), fires shot n+1 with the same options and time stepping as run_w_multi.py, and
records the surface temperature, melt-front depth and interface temperature at every time step.
From that it reports the melt duration (total time with a molten layer) and the maximum melt
depth of the shot.

Usage: python scripts/run_extra_shot.py <run_dir> <scenario> [option=value ...]
  option=value must repeat the options of the multi-shot run (e.g. k_irr=0.5).
Output: <run_dir>_extra_shot/<scenario>_extra_shot.json, _history.csv, _maxT_profile.csv,
        _He_layer.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall.diagnostics import ShotRecorder  # noqa: E402
from ife_wall.scenarios import SCENARIOS, interface_index, w_case  # noqa: E402
from ife_wall.thermal import EnergyLedger  # noqa: E402
from run_w_single import k_status, parse_overrides, save_diagnostics  # noqa: E402

run_dir, name = Path(sys.argv[1]), sys.argv[2]
opt = parse_overrides(sys.argv[3:])
scen = SCENARIOS[name]
P = 1.0 / scen.rep_Hz
z = np.load(run_dir / f"{name}_checkpoint.npz", allow_pickle=False)
rows = z["rows"]
k = len(rows)  # index of the extra shot (0-based), i.e. shot number k + 1
s = w_case(scen, opt, n_shots=k + 1)
s.m, s.H, s.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
s.m_target0 = z["m_target0"].copy()
s.t, s.mass_evaporated, s.n_rezones = float(z["t"]), float(z["mass_evaporated"]), int(z["n_rezones"])
s.ledger = EnergyLedger(**json.loads(str(z["ledger"])))
s._frac_cache = None
assert abs(s.t - k * P) < 1e-9, f"checkpoint time {s.t} does not match shot {k}"

out = run_dir.parent / f"{run_dir.name}_extra_shot"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
pre_Ts = s.surface_temperature()
abl0 = s.mass_evaporated
recd = ShotRecorder(s, s.t, P)
hist = []


def rec(sv):
    hist.append((sv.t - k * P, sv.surface_temperature(), sv.melt_depth(), float(sv.T[interface_index(sv)])))
    recd.update(sv)


s.run(k * P + 50e-6, dt0=1e-13, dT_target=5.0, record=rec)
s.run((k + 1) * P, dt0=1e-9, dT_target=5.0, record=rec)
h = np.array(hist)
dt = np.diff(np.concatenate([[0.0], h[:, 0]]))
molten = h[:, 2] > 0
melt_dur = float(np.sum(dt[molten]))  # step durations ending with a molten layer
i_first = int(np.argmax(molten)) if molten.any() else None
i_last = int(len(molten) - 1 - np.argmax(molten[::-1])) if molten.any() else None
ks, sks = k_status(opt)
res = {"source_checkpoint": str(run_dir / f"{name}_checkpoint.npz"), "scenario": name, "shot_number": k + 1,
       "k_irr": opt.k_irr, "skin_um": opt.skin_um, "skin_factor": opt.skin_factor, "k_status": ks, "skin_status": sks,
       "preshot_surface_T_K": pre_Ts, "peak_surface_T_K": float(h[:, 1].max()),
       "t_peak_us": float(h[np.argmax(h[:, 1]), 0] * 1e6),
       "max_melt_depth_um": float(h[:, 2].max() * 1e6), "melt_duration_us": melt_dur * 1e6,
       "melt_start_us": float(h[i_first, 0] * 1e6) if molten.any() else None,
       "melt_end_us": float(h[i_last, 0] * 1e6) if molten.any() else None,
       "peak_interface_T_K": float(h[:, 3].max()), "ablated_nm_this_shot": (s.mass_evaporated - abl0) / 19250 * 1e9,
       "note": "melt duration = total time with a molten layer at the surface (sum of time steps ending molten); "
               "the melt front is the solver's melt_depth()",
       "diagnostics": recd.results(), "wall_s": time.time() - t0}
np.savetxt(out / f"{name}_extra_shot_history.csv", np.column_stack([h[:, 0] * 1e6, h[:, 1], h[:, 2] * 1e6, h[:, 3]]),
           delimiter=",", header="t_since_shot_us,surface_T_K,melt_depth_um,interface_T_K", comments="")
save_diagnostics(out, name, {"_maxT_profile": recd.profile_rows(), "diagnostics": res["diagnostics"]})
(out / f"{name}_extra_shot.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(f"{name} extra shot {k + 1}: peak {res['peak_surface_T_K']:.0f} K, melt max {res['max_melt_depth_um']:.3f} um, "
      f"melt duration {res['melt_duration_us']:.2f} us, {res['wall_s']:.0f}s")
