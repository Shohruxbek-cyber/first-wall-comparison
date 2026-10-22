"""Repeated-shot minimum lithium film for the variants of the revision of 20 Oct 2026: neutron heating
(steel factor 0.5 and 1; results/neutron/li_multi_Demo_d<delta>um_steel<f>/) and vapor shielding
(f_rr = 0, s = 1; results/vapor_shielding/multi/Demo_d<delta>um_frr0_s1/), next to the production
runs without either (runs/li_multi/Demo_d<delta>um/).

Same rule as scripts/li_multi_summary.py (crossing_delta): the maximum peak steel temperature over the
run, interpolated linearly in temperature and logarithmically in thickness between the last failing
and the first passing thickness; also the 873 K lithium/steel compatibility limit applied to the
peak, as in scripts/diag_li_compatibility.py; and the lithium evaporated in the last shot, interpolated
to the minimum film in the same way.  Only finished runs count (a Demo.json written at the end of the
run); a bracket whose runs have not all converged is flagged.

Usage: python scripts/diag_li_multi_variants.py [--out=results/revision/li_multi_variants]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "production (no neutron heating, no vapor shielding)": (ROOT / "runs" / "li_multi", r"Demo_d([0-9.]+)um$"),
    "neutron heating, steel factor 0.5": (ROOT / "results" / "neutron", r"li_multi_Demo_d([0-9.]+)um_steel0\.5$"),
    "neutron heating, steel factor 1": (ROOT / "results" / "neutron", r"li_multi_Demo_d([0-9.]+)um_steel1$"),
    "vapor shielding, f_rr = 0, s = 1": (ROOT / "results" / "vapor_shielding" / "multi", r"Demo_d([0-9.]+)um_frr0_s1$"),
}
LIMITS = (925.0, 873.0)


def crossing_delta(rows, limit):
    """scripts/li_multi_summary.py, crossing_delta, with 'pass' evaluated for the given limit: the
    thickness where the run's maximum peak steel T crosses the limit, with its grid bracket; if every
    run passes, the thinnest one (an upper bound); if none passes, None."""
    rows = sorted(({**r, "pass": r["peak_K"] <= limit and r["status"] == "ok"} for r in rows), key=lambda r: r["delta_um"])
    for a, b in zip(rows[:-1], rows[1:]):
        if (not a["pass"]) and b["pass"]:
            Ta, Tb = a["peak_K"], b["peak_K"]
            if a["status"] != "ok" or Ta <= Tb:
                return b["delta_um"], (a["delta_um"], b["delta_um"])
            f = (Ta - limit) / (Ta - Tb)
            return float(np.exp(np.log(a["delta_um"]) + f * np.log(b["delta_um"] / a["delta_um"]))), (a["delta_um"], b["delta_um"])
    if rows and rows[0]["pass"]:
        return rows[0]["delta_um"], (None, rows[0]["delta_um"])
    return None, (None, None)


def at_film(rows, d, bracket, key):
    """`key` of the bracket's runs interpolated to the thickness d as the film itself (linearly in log
    thickness); the passing run's value when d is that run's thickness or the bracket is open."""
    a, b = bracket
    if d is None:
        return None
    rb = next(q for q in rows if q["delta_um"] == b)
    if a is None or d == b:
        return rb[key]
    ra = next(q for q in rows if q["delta_um"] == a)
    f = np.log(d / a) / np.log(b / a)
    return float(ra[key] + f * (rb[key] - ra[key]))


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_multi_variants"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (base, pat) in GROUPS.items():
        rows = []
        for p in sorted(base.glob("*")):
            m = re.match(pat, p.name)
            if not m or not (p / "Demo.json").exists() or not (p / "Demo_per_shot.csv").exists():
                continue
            j = json.loads((p / "Demo.json").read_text(encoding="utf-8"))
            r = np.loadtxt(p / "Demo_per_shot.csv", delimiter=",", skiprows=1, ndmin=2)
            if len(r) != int(j["n_shots"]):  # Demo.json is written every 10 shots; count only a finished run
                continue
            finished = bool(j.get("converged")) or int(j["n_shots"]) >= int(j["max_n_shots"]) or j.get("status") != "ok"
            if not finished:
                continue
            rows.append({"delta_um": j["delta_um"], "folder": p.relative_to(ROOT).as_posix(), "n_shots": int(j["n_shots"]),
                         "converged": bool(j["converged"]), "status": j["status"],
                         "peak_K": float(r[:, 2].max()), "last_shot_peak_K": float(r[-1, 2]),
                         "last_preshot_front_K": float(r[-1, 1]),
                         "vaporized_last_shot_kg_m2": float(r[-1, 5]), "energy_rel_residual": j["energy_rel_residual"]})
        rows.sort(key=lambda q: q["delta_um"])
        g = {"rows": rows}
        for lim in LIMITS:
            d, br = crossing_delta(rows, lim)
            conv = all(q["converged"] for q in rows if q["delta_um"] in br)
            g[f"min_film_um_{lim:.0f}K"] = d
            g[f"bracket_um_{lim:.0f}K"] = br
            g[f"bracket_converged_{lim:.0f}K"] = conv
            g[f"vaporized_last_shot_at_min_film_kg_m2_{lim:.0f}K"] = at_film(rows, d, br, "vaporized_last_shot_kg_m2")
        res[name] = g
        print(name)
        for q in rows:
            print(f"  {q['delta_um']:7.2f} um: {q['n_shots']} shots, converged {q['converged']}, max peak steel {q['peak_K']:.1f} K, "
                  f"pre-shot front {q['last_preshot_front_K']:.1f} K")
        for lim in LIMITS:
            d = g[f"min_film_um_{lim:.0f}K"]
            v = g[f"vaporized_last_shot_at_min_film_kg_m2_{lim:.0f}K"]
            print(f"  {lim:.0f} K: {'—' if d is None else f'{d:.2f} um'} (bracket {g[f'bracket_um_{lim:.0f}K']}, "
                  f"converged {g[f'bracket_converged_{lim:.0f}K']}); evaporated per shot there "
                  f"{'—' if v is None else f'{v:.4g} kg m-2'}")
    (out / "li_multi_variants.json").write_text(json.dumps({
        "note": "repeated-shot minimum Li film (Demo, 10 Hz, T_Li = 600 K, coolant 600 K, h = 10 kW m-2 K-1) for the variants of "
                "20 Oct 2026; rule as scripts/li_multi_summary.py; 873 K = Li/steel compatibility limit applied to the peak",
        "groups": res}, indent=1), encoding="utf-8")
