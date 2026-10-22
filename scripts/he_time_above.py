"""Time above a temperature threshold, from the RECORDED T(x,t) of the Demo quasi-steady cycle
(runs/he_retention/thermal/Demo_T<converged pre-shot T>_ss.npz, written by scripts/record_he_thermal.py).

Linear interpolation in time between recorded steps (the steps are ~9 ns apart around the peak) and in depth between the stored
depths.  Prints and writes runs/he_retention/results/time_above_threshold_Demo.json.
Sanity checks printed first: time axis in seconds, 0 ... 0.1 s, strictly increasing; peak surface temperature (the quasi-steady
Demo peak of the thermal code) at ~2 us.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import converged_preshot  # noqa: E402
z = np.load(ROOT / f"runs/he_retention/thermal/Demo_T{converged_preshot('Demo', legacy=921.1):.1f}_ss.npz")
t, zz, T = z["t"], z["z"], z["T"].astype(float)
assert t[0] == 0.0 and abs(t[-1] - 0.1) < 1e-12 and np.all(np.diff(t) > 0), "time axis is not 0 ... 0.1 s in seconds"
print(f"{len(t)} recorded steps, t = {t[0]:g} ... {t[-1]:g} s, strictly increasing; depth grid {zz[0]:g} ... {zz[-1]:g} m ({len(zz)} points)")
print(f"peak surface T {T[:, 0].max():.1f} K at {t[np.argmax(T[:, 0])] * 1e6:.2f} us; pre-shot {T[0, 0]:.1f} K, end of cycle {T[-1, 0]:.1f} K")


def time_above(Ts, thr):
    segs, inside, start = [], bool(Ts[0] >= thr), (t[0] if Ts[0] >= thr else None)
    for k in range(len(t) - 1):
        a, b = Ts[k], Ts[k + 1]
        if (a >= thr) != (b >= thr):
            tc = t[k] + (thr - a) / (b - a) * (t[k + 1] - t[k])
            if not inside:
                start, inside = tc, True
            else:
                segs.append((start, tc))
                inside = False
    if inside:
        segs.append((start, t[-1]))
    return sum(e - s for s, e in segs), segs


out = {}
for d_um in (0, 1, 3, 5, 10):
    Ts = np.array([np.interp(d_um * 1e-6, zz, T[k]) for k in range(len(t))])
    row = dict(peak_K=float(Ts.max()), t_peak_us=float(t[np.argmax(Ts)] * 1e6))
    for thr in (2000.0, 1500.0, 1000.0):
        tt, segs = time_above(Ts, thr)
        row[f"time_above_{int(thr)}K_us"] = float(tt * 1e6)
        row[f"window_above_{int(thr)}K_us"] = [[float(s * 1e6), float(e * 1e6)] for s, e in segs]
    out[f"{d_um}um"] = row
    print(f"depth {d_um:>2} um: peak {row['peak_K']:.0f} K at {row['t_peak_us']:.2f} us; time above 2000 K = {row['time_above_2000K_us']:.2f} us, "
          f"above 1500 K = {row['time_above_1500K_us']:.2f} us, above 1000 K = {row['time_above_1000K_us']:.0f} us")
(ROOT / "runs/he_retention/results").mkdir(parents=True, exist_ok=True)
(ROOT / "runs/he_retention/results/time_above_threshold_Demo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
