"""Mean stopping depth of the implanted helium, burn and debris (revision of 21 Oct 2026, final number check).

Appendix I.4 of the article quotes the mean stopping depths of the burn and the debris helium in Demo;
this script computes them with the production helium sources (ife_wall.he.sources.ShotSources on the
production grid of scripts/run_he_retention.py): the mean of the cell centers weighted with the helium
that each sub-bin implants in each cell, and the implanted total per shot as a check (Appendix I.4).
Runs no solver.  Usage: python scripts/diag_he_depths.py [--out=results/revision/he_depths]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.sources import ShotSources  # noqa: E402

if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "he_depths"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for scen in ("Experimental", "Prototype", "Demo"):
        src = ShotSources(scen, make_grid())
        r = {}
        for g in ("burn", "debris", "all"):
            n_impl = depth_sum = within_5um = 0.0
            for i in src.he_idx:
                s = src.sub[i]
                if g != "all" and s.group != g:
                    continue
                f = src.he_implant_fraction(s.species, s.E) * s.n  # He per m^2 stopping in each cell
                n_impl += float(f.sum())
                depth_sum += float((f * src.x).sum())
                within_5um += float(f[src.edges[1:] <= 5e-6].sum())
            r[g] = {"implanted_He_per_m2_per_shot": n_impl, "mean_stopping_depth_um": depth_sum / n_impl * 1e6,
                    "fraction_within_5um": within_5um / n_impl}
        r["incident_He_per_m2_per_shot"] = float(sum(src.sub[i].n for i in src.he_idx))
        res[scen] = r
        print(scen, json.dumps(r))
    (out / "he_depths.json").write_text(json.dumps({
        "note": "mean stopping depth of the implanted helium (production helium sources, production grid), weighted "
                "with the implanted number per cell", "scenarios": res}, indent=1), encoding="utf-8")
