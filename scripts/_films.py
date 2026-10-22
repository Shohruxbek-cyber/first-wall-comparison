"""Film thicknesses of the current results (revision of 22 Oct 2026): the minimum films, the direct-run films and the
helium-exclusion films, read from the summaries of the finished runs, for the scripts that evaluate quantities at
those films (steel_at_films.py --auto, diag_film_travel.py).  Nothing is computed here except the range in lithium
of a burn alpha at the top of its 0.5 MeV bins (the top of the main burn-alpha population)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def direct_film_um(d_um):
    """Thickness of the direct run at an interpolated minimum: the nearest whole micrometer (as for 45.93 -> 46)."""
    return float(round(d_um))


def burn_alpha_top_range_um():
    import numpy as np
    from ife_wall import srim
    from ife_wall.ions import KEV_J, ion_spectra_tables
    from ife_wall.library import lithium
    top_keV = float(ion_spectra_tables("conserving")[("burn", "4He")]["hi"][7])  # 4.0 MeV
    rho = float(lithium().rho(np.array(600.0)))
    return top_keV, float(srim.SrimStopping(mode="projected").range("4He", "Li", top_keV * KEV_J)) / rho * 1e6


def current_films():
    """{scenario: {label: film in um}} from the stored summaries (each must exist)."""
    comp = _load("results/revision/li_compatibility/li_compatibility.json")
    md = _load("results/revision/li_direct/minima_direct.json")["minima"]
    crit = _load("results/revision/steel_behind_film/steel_criteria.json")["scenarios"]
    summ = _load("runs/summary/summary.json")["Li"]
    f = {"Experimental": {"925 K, single shot": summ["Experimental"]["min_delta_um"]},
         "Prototype": {"925 K, single shot": comp["Prototype"]["single_shot_min_film_um"]["925 K"],
                       "925 K, repeated shots": comp["Prototype"]["repeated_shot_min_film_um"]["925 K"],
                       "873 K, single shot": comp["Prototype"]["single_shot_min_film_um"]["873 K"],
                       "873 K, repeated shots": comp["Prototype"]["repeated_shot_min_film_um"]["873 K"]},
         "Demo": {"925 K, single shot": comp["Demo"]["single_shot_min_film_um"]["925 K"],
                  "873 K, single shot": comp["Demo"]["single_shot_min_film_um"]["873 K"]}}
    d = f["Demo"]
    for lv in ("925", "873"):
        c = md["production_and_direct"][f"{lv}K"]
        if c:
            d[f"{lv} K, repeated shots (with the direct runs)"] = c["delta_um"]
        g = md["production_only"][f"{lv}K"]
        if g:
            d[f"{lv} K, film of the direct run"] = direct_film_um(g["delta_um"])
    for v, lab in (("neutron_steel0.5", "neutron heating 0.5"), ("neutron_steel1", "neutron heating 1")):
        c = md.get(f"{v}_with_direct", {}).get("925K")
        if c:
            d[f"925 K, repeated shots, {lab} (with the direct runs)"] = c["delta_um"]
    top_keV, r_um = burn_alpha_top_range_um()
    d[f"range of a {top_keV / 1e3:g} MeV burn alpha (top of the main burn-alpha population)"] = r_um
    d["helium, 10^6 shots to 10^19 m^-2"] = crit["Demo"]["film_um"]["1e6_shots_to_1e19"]
    d["helium, 10^6 shots to 10^18 m^-2"] = crit["Demo"]["film_um"]["1e6_shots_to_1e18"]
    d["helium, direct run film"] = direct_film_um(crit["Demo"]["film_um"]["1e6_shots_to_1e18"])
    return f
