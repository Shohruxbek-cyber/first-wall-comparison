"""Helium, hydrogen isotopes and displacement damage in the steel behind lithium films of given thickness
(revision of 21 Oct 2026): the minimum films of the thermal criteria and the helium-exclusion
films, with ife_wall.he.steel.SteelBehindFilm exactly as in scripts/steel_behind_film.py (production grid of
2400 cells over 120 um of steel, 24 sub-bins per energy bin).

Usage: python scripts/steel_at_films.py <Scenario>:<um>[,<um>...] [<Scenario>:...] [--out=DIR]
       python scripts/steel_at_films.py --auto [--out=DIR]   the films of the current results (scripts/_films.py:
              minima, direct-run and helium-exclusion films, the top of the burn-alpha population), labelled
       (default DIR: results/revision/steel_behind_film; output steel_at_films.json)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.he.sources import split_bins  # noqa: E402
from ife_wall.he.steel import SteelBehindFilm  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "steel_behind_film"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    req, labels = {}, {}
    if "--auto" in sys.argv[1:]:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _films import current_films
        for name, films in current_films().items():
            req[name] = list(films.values())
            labels[name] = list(films.keys())
    for a in sys.argv[1:]:
        if a.startswith("--"):
            continue
        name, films = a.split(":", 1)
        req[name] = [float(x) for x in films.split(",")]
    res = {"note": __doc__.split("Usage:")[0].strip(), "requested_um": req, "labels": labels, "scenarios": {}}
    for name, films in req.items():
        sub = split_bins(SCENARIOS[name], 24)
        res["scenarios"][name] = [SteelBehindFilm(name, d * 1e-6, sub_bins=sub).summary() for d in films]
        for r, lab in zip(res["scenarios"][name], labels.get(name, [])):
            r["label"] = lab
        for r in res["scenarios"][name]:
            he = r["He_per_m2_per_shot"]
            print(f"{name} {r['delta_um']:8.3f} um: He {he['burn']:.3e} + {he['debris']:.3e} m-2/shot, "
                  f"H {r['H_isotopes_per_m2_per_shot']['total']:.3e}, dpa peak {r['dpa_per_shot_peak']:.3e} "
                  f"mean 0-5 um {r['dpa_per_shot_mean_0_5um']:.3e}, shots to 1e18 / 1e19: "
                  f"{r['shots_to_1e18_He_m2'] or 0:.4g} / {r['shots_to_1e19_He_m2'] or 0:.4g}")
    (out / "steel_at_films.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
