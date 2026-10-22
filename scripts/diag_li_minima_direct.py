"""Repeated-shot minimum lithium films with the direct runs of 21 Oct 2026.

The minimum film is the thickness at which the peak steel temperature of the quasi-steady shot crosses the limit,
interpolated linearly in temperature and logarithmically in film thickness between the two computed films that
bracket it (Section 2.7).  The points are the converged repeated-shot runs of the production scan
(runs/li_multi/Demo_d*um) and the direct runs at 46, 58 and 180 um (results/revision/li_direct); the same rule is
applied to the bounding runs with the latent heat of the vapor returned to the film
(results/revision/li_return_latent).  Limits: 925 K (structural) and 873 K (Li/steel compatibility).
Since 22 Oct 2026 also the variants, each with its own runs and its direct runs
at the thicknesses interpolated from them (results/revision/li_direct_variants/<variant>/Demo_d*um): steel neutron
heating at 0.5 and 1 times the tungsten value, vapor shielding (f_rr = 0, s = 1) and the coolant 3.5 mm behind the
surface.

Usage: python scripts/diag_li_minima_direct.py [--out=results/revision/li_direct]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMITS = (925.0, 873.0)
VARIANTS = {  # name: (glob of the variant's own runs, relative to ROOT)
    "neutron_steel0.5": "results/neutron/li_multi_Demo_d*um_steel0.5",
    "neutron_steel1": "results/neutron/li_multi_Demo_d*um_steel1",
    "vapor_frr0_s1": "results/vapor_shielding/multi/Demo_d*um_frr0_s1",
    "steel_3.5mm": "results/cooling_3p5mm/li_multi_Demo_d*um",
}


def points(dirs):
    pts = {}
    for d in dirs:
        f = d / "Demo.json"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            pts[float(j["delta_um"])] = {"peak_K": float(j["last_shot_peak_steel_K"]), "converged": bool(j.get("converged")),
                                         "evap_kg_m2": float(j["last_shot_vaporized_kg_m2"]),
                                         "n_shots": int(j.get("n_shots", 0)), "source": str(d.relative_to(ROOT)).replace("\\", "/")}
    return dict(sorted(pts.items()))


def crossing(pts, limit):
    ds = list(pts)
    for a, b in zip(ds, ds[1:]):
        Ta, Tb = pts[a]["peak_K"], pts[b]["peak_K"]
        if Ta > limit >= Tb:
            f = (Ta - limit) / (Ta - Tb)
            ea, eb = pts[a]["evap_kg_m2"], pts[b]["evap_kg_m2"]
            return {"delta_um": math.exp(math.log(a) + f * (math.log(b) - math.log(a))), "bracket_um": [a, b], "bracket_K": [Ta, Tb],
                    "evap_kg_m2_at_minimum": ea + f * (eb - ea)}
    return None


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_direct"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    prod = sorted((ROOT / "runs" / "li_multi").glob("Demo_d*um"))
    direct = sorted((ROOT / "results" / "revision" / "li_direct").glob("Demo_d*um"))
    rl = sorted((ROOT / "results" / "revision" / "li_return_latent").glob("Demo_d*um"))
    sets = {"production_only": points(prod), "production_and_direct": points(prod + direct), "return_latent": points(rl)}
    for v, pattern in VARIANTS.items():
        own = sorted(ROOT.glob(pattern))
        vd = sorted((ROOT / "results" / "revision" / "li_direct_variants" / v).glob("Demo_d*um"))
        sets[f"{v}_own_runs"] = points(own)
        sets[f"{v}_with_direct"] = points(own + vd)
    res = {"note": __doc__.split("Usage:")[0].strip(), "points": sets,
           "minima": {k: {f"{lv:.0f}K": crossing(p, lv) for lv in LIMITS} for k, p in sets.items()}}
    (out / "minima_direct.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, p in sets.items():
        print(k, {d: (round(v["peak_K"], 1), v["converged"]) for d, v in p.items()})
        for lv in LIMITS:
            c = res["minima"][k][f"{lv:.0f}K"]
            print(f"   {lv:.0f} K: " + (f"{c['delta_um']:.2f} um between {c['bracket_um']}" if c else "not bracketed"))


if __name__ == "__main__":
    main()
