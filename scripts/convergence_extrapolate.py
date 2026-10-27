"""Discretization error of the production settings, from the convergence study (revision of
20 Oct 2026): reads results/revision/convergence/convergence_summary.json and, for the growth
factor (g - 1 halved: 1.10 -> 1.05 -> 1.025) and the time-step limits (halved and doubled),
estimates the error of the baseline by Richardson's rule with the observed order p,
    e = (f_base - f_fine) / (1 - 2**-p),
and the extrapolated value f_base - e_growth - e_time (the two errors taken as additive).  The
surface-cell refinement changes every quantity by less than 0.04 K or 0.03 % without a
systematic trend and is not extrapolated.  An observed order outside 0.5-3, or one obtained from
non-monotonic changes (the halved and the doubled setting moving the quantity the same way), is
replaced by the nominal order of the scheme (1 in time, 2 in space) and flagged.  For the quasi-steady tungsten
cycle the grid settings start from a state interpolated onto the new cells, which shifts the
pre-shot and interface temperatures of all of them by 0.03-0.06 K; their growth estimate is
therefore flagged as an upper bound.

Usage: python scripts/convergence_extrapolate.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "revision" / "convergence"
NOMINAL = {"growth": 2.0, "time step": 1.0}
FINE = {"growth": "growth_1.025", "time step": "dt_half"}
COARSE = {"growth": "growth_1.10", "time step": "dt_double"}

if __name__ == "__main__":
    summ = json.loads((D / "convergence_summary.json").read_text(encoding="utf-8"))
    out = {}
    for q, v in summ.items():
        vals, orders = v["values"], v["observed_order"]
        base = vals["baseline"]
        res = {"baseline": base, "errors": {}}
        total = 0.0
        for kind, fine in FINE.items():
            p = orders.get(kind)
            flag = None
            if (vals[COARSE[kind]] - base) * (base - vals[fine]) < 0:
                flag = f"non-monotonic changes: observed order {p} replaced by the nominal {NOMINAL[kind]}"
                p = NOMINAL[kind]
            elif p is None or not 0.5 <= p <= 3.0:
                flag = f"observed order {p} replaced by the nominal {NOMINAL[kind]}"
                p = NOMINAL[kind]
            e = (base - vals[fine]) / (1.0 - 2.0 ** -p)
            if q.startswith("W cycle") and kind == "growth":
                flag = (flag + "; " if flag else "") + "grid settings start from an interpolated state (upper bound)"
            res["errors"][kind] = {"order_used": p, "error_of_baseline": e, "flag": flag}
            total += e
        res["extrapolated"] = base - total
        res["extrapolated_minus_baseline"] = -total
        out[q] = res
    (D / "convergence_extrapolated.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for q, r in out.items():
        print(f"{q}: baseline {r['baseline']:.6g}, extrapolated {r['extrapolated']:.6g} "
              f"({r['extrapolated_minus_baseline']:+.3g}); "
              + ", ".join(f"{k} p={e['order_used']:.2f} e={e['error_of_baseline']:+.3g}" for k, e in r["errors"].items()))
