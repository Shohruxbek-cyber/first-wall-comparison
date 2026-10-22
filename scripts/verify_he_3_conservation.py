"""Verification test 3: He conservation over many shots.
He implanted (integrated source, carried as a state variable) = He mobile + He trapped (+ He clusters, Level 2)
+ He released through the surface.  The back face of the armour is closed (no flux) so nothing leaves there.
Pass: relative error < 1e-10 at every shot.  Checked (a) in a fresh run here (Demo baseline and Level 2,
40 shots each) and (b) in the long baseline runs, if present (runs/he_retention/<scenario>_baseline/totals.csv).
"""
import sys

import numpy as np

from _he_common import ROOT, record, style, VER
from ife_wall.scenarios import converged_preshot  # noqa: E402
from ife_wall.he import cases as CS
from ife_wall.he.simulate import FastSimulation


def fresh(case, nshots=40):
    params, kw = CS.build(case)
    sim = FastSimulation("Demo", params, str(ROOT / f"runs/he_retention/thermal/Demo_T{converged_preshot('Demo', legacy=921.1):.1f}_ss.npz"), **kw)
    errs, ret = [], []
    for _ in range(nshots):
        sim.shot()
        sn = sim.snapshot()
        errs.append(abs(sn["He_cons_err"]) / sn["He_in"])
        ret.append(sn["He_retained"])
    return np.array(errs), np.array(ret)


def main():
    plt = style()
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for case in ("baseline", "level2"):
        e, r = fresh(case)
        out[f"fresh_{case}_max_rel_err_40_shots"] = float(e.max())
        ax.semilogy(np.arange(1, len(e) + 1), np.maximum(e, 1e-17), label=f"fresh run, {case}")
    for scen in ("Demo", "Prototype"):   # Experimental is a single-shot design: not run as a multi-shot case
        f = ROOT / "runs" / "he_retention" / f"{scen}_baseline" / "totals.csv"
        if f.exists():
            d = np.genfromtxt(f, delimiter=",", names=True)
            e = np.abs(d["He_cons_err"]) / d["He_in"]
            out[f"long_{scen}_baseline_max_rel_err_{len(e)}_shots"] = float(e.max())
            ax.semilogy(d["shot"], np.maximum(e, 1e-17), lw=1, label=f"{scen} baseline ({len(e)} shots)")
    ax.axhline(1e-10, color="k", ls="--"); ax.text(2, 1.5e-10, "threshold 1e-10", fontsize=10)
    ax.set(xlabel="shot number", ylabel="|He balance error| / He implanted", ylim=(1e-17, 1e-8))
    ax.legend(fontsize=9)
    fig.savefig(str(VER / "test3_conservation.pdf"))
    worst = max(v for v in out.values())
    record("test3_conservation", worst < 1e-10, max_relative_error=worst, threshold=1e-10, details=out,
           note="V and I balances are not part of the compiled ledger; they are checked against the NumPy reference model "
                "(scripts/verify_he_3b_numpy_crosscheck.py)")


if __name__ == "__main__":
    main()
