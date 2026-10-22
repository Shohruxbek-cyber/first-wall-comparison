"""Verification test 5: convergence.  Retained He (and its depth distribution) after N = 30 Demo shots changes by
< 2 % when (a) the grid spacing is halved (every cell split in two), (b) the time tolerance is tightened from
rtol = 1e-3 to 1e-6 (atol 1e-9 -> 1e-11), (c) the source discretisation is refined (sub-bins 6 -> 12, time slices
40 -> 80).
"""
import sys
import time

import numpy as np

from _he_common import ROOT, record, style, VER
from ife_wall.scenarios import converged_preshot  # noqa: E402
from ife_wall.he import cases as CS
from ife_wall.he.grid import make_grid, refine
from ife_wall.he.simulate import FastSimulation

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CYC = str(ROOT / f"runs/he_retention/thermal/Demo_T{converged_preshot('Demo', legacy=921.1):.1f}_ss.npz")


def run(label, edges=None, **kw):
    params, sim_kw = CS.build("baseline")
    sim_kw.update(kw)
    t0 = time.time()
    sim = FastSimulation("Demo", params, CYC, edges=edges if edges is not None else make_grid(), **sim_kw)
    for _ in range(N):
        sim.shot()
    sn = sim.snapshot()
    pr = sim.fm.profiles(sim.y)
    x = pr["x"] * 1e6
    w = sim.fm.dx
    bins = [(0, 1), (1, 5), (5, 60)]
    dist = [float(((pr["He_trapped"] + pr["He"])[(x >= a) & (x < b)] * w[(x >= a) & (x < b)]).sum()) for a, b in bins]
    print(f"{label}: retained {sn['He_retained']:.5e}  released {sn['He_surface']:.5e}  ({time.time() - t0:.0f}s)", flush=True)
    return sn, dist, x, pr


def main():
    plt = style()
    ref = run("reference (production grid, rtol 1e-3)")
    runs = {
        "grid halved": run("grid halved", edges=refine(make_grid(), 2)),
        "tolerance tightened": run("rtol 1e-6, atol 1e-11", rtol=1e-6, atol=1e-11),
        "sources refined": run("sub-bins 12, slices 80", n_sub=12, n_slices=80),
    }
    out, worst = {}, 0.0
    for k, (sn, dist, x, pr) in runs.items():
        d_ret = abs(sn["He_retained"] - ref[0]["He_retained"]) / ref[0]["He_retained"]
        d_rel = abs(sn["He_surface"] - ref[0]["He_surface"]) / ref[0]["He_surface"]
        d_dist = [abs(a - b) / b for a, b in zip(dist, ref[1])]
        out[k] = dict(retained=sn["He_retained"], rel_change_retained=d_ret, rel_change_released=d_rel, rel_change_by_depth_bin=d_dist)
        worst = max(worst, d_ret)
    out["reference"] = dict(retained=ref[0]["He_retained"], released=ref[0]["He_surface"], by_depth_bin_0_1_5_60um=ref[1])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xr = ref[2]; ax.loglog(xr, ref[3]["He_trapped"], "k-", lw=2, label="reference")
    for (k, (sn, dist, x, pr)), c in zip(runs.items(), ["C0", "C1", "C3"]):
        ax.loglog(x, pr["He_trapped"], "--", color=c, label=k)
    ax.set(xlabel="depth ($\\mu$m)", ylabel="trapped He (appm)", title=f"Test 5: trapped He after {N} shots", xlim=(1e-3, 100), ylim=(1e-3, 1e2))
    ax.legend(fontsize=9)
    fig.savefig(str(VER / "test5_convergence.pdf"))
    record("test5_convergence", worst < 0.02, max_change_retained=worst, threshold=0.02, N_shots=N, details=out)


if __name__ == "__main__":
    main()
