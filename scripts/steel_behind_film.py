"""Helium, hydrogen isotopes and displacement damage in the steel behind the lithium film (revision of
21 Oct 2026), with ife_wall.he.steel.

For the three scenarios and the 13 film thicknesses of the lithium scan (and a fine grid for the criteria):
the helium implanted in the steel per shot (burn and debris), the hydrogen-isotope fluence, the ion energy
carried into the steel, the dpa per shot (peak and mean over the first 5 um of steel, NRT, Ed = 40 eV), and the
shots to 1e18 and 1e19 He m^-2 with all helium retained, as for tungsten.  Criteria: the film that stops the
main burn-alpha population (the 4He bins up to 3.54 MeV, at the upper edge of the 3.54 MeV bin) and the films
beyond which the steel helium count exceeds 1e6 shots (for 1e18 and 1e19 m^-2).  Check: the ion energy into the
steel against the deposition profiles of the production model (runs/deposition/li_film_absorption.json).

Usage: python scripts/steel_behind_film.py [--out=results/revision/steel_behind_film]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.he.sources import HE_SPECIES, split_bins  # noqa: E402
from ife_wall.he.steel import SteelBehindFilm  # noqa: E402
from ife_wall.ions import load_ion_bins  # noqa: E402
from ife_wall.library import lithium  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

GRID_UM = [1.0, 1.778, 3.162, 5.623, 10.0, 17.783, 31.623, 56.234, 100.0, 177.828, 316.228, 562.341, 1000.0]
SCEN = ("Experimental", "Prototype", "Demo")


def crossing(x, y, level):
    """First x (log interpolation) at which y falls below level, for y decreasing with x; None if never."""
    for i in range(1, len(x)):
        if y[i - 1] > level >= y[i]:
            if y[i] <= 0:
                return float(x[i])
            f = (np.log(level) - np.log(y[i - 1])) / (np.log(y[i]) - np.log(y[i - 1]))
            return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1]))))
    return None


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "steel_behind_film"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rho = float(lithium().rho(np.array(600.0)))
    st = srim.SrimStopping(mode="projected")
    # main burn-alpha population: 4He bins up to the 3.54 MeV bin; its upper edge
    burn_he = [b for b in load_ion_bins(True) if b.group == "burn" and b.species == "4He" and b.number > 0]
    main = [b for b in burn_he if b.E_mean <= 3.6e6 * 1.602176634e-19]
    E_top = max(b.E_hi for b in main)
    d_alpha_main_um = float(st.range("4He", "Li", E_top)) / rho * 1e6
    res = {"note": __doc__.split("Usage:")[0].strip(), "Li_density_kg_m3_at_600K": rho,
           "main_burn_alpha_top_energy_MeV": E_top / 1.602176634e-13, "film_stopping_main_burn_alphas_um": d_alpha_main_um,
           "scenarios": {}}
    dep = {round(r["delta_um"], 3): r for r in json.loads((ROOT / "runs" / "deposition" / "li_film_absorption.json")
                                                          .read_text(encoding="utf-8"))}
    fine = np.geomspace(1.0, 3000.0, 241)
    for name in SCEN:
        sub = split_bins(SCENARIOS[name], 24)
        incident_ion = sum(s.n * s.E for s in sub)
        rows = []
        for d in GRID_UM:
            s = SteelBehindFilm(name, d * 1e-6, sub_bins=sub).summary()
            s["ion_energy_fraction_into_steel"] = s["ion_energy_into_steel_J_m2"] / incident_ion
            if name == "Demo" and round(d, 3) in dep:
                s["production_ion_in_Fe_fraction"] = dep[round(d, 3)]["ion_in_Fe"]
            rows.append(s)
        he_fine, he_burn_fine = [], []
        for d in fine:
            sb = SteelBehindFilm(name, d * 1e-6, sub_bins=sub, n_cells=10)
            he_fine.append(sb.fluence(HE_SPECIES))
            he_burn_fine.append(sb.fluence(HE_SPECIES, "burn"))
        he_fine = np.array(he_fine)
        res["scenarios"][name] = {
            "incident_ion_energy_J_m2": incident_ion,
            "incident_He_per_m2": {"burn": sum(s.n for s in sub if s.species in HE_SPECIES and s.group == "burn"),
                                   "debris": sum(s.n for s in sub if s.species in HE_SPECIES and s.group == "debris")},
            "rows": rows,
            "film_for_1e6_shots_to_1e18_um": crossing(fine, he_fine, 1e18 / 1e6),
            "film_for_1e6_shots_to_1e19_um": crossing(fine, he_fine, 1e19 / 1e6),
            "film_stopping_all_burn_He_um": float(fine[np.argmax(np.array(he_burn_fine) <= 0)]) if (np.array(he_burn_fine) <= 0).any() else None,
            "fine_grid": {"delta_um": fine.tolist(), "He_per_m2_per_shot": he_fine.tolist(), "burn_He_per_m2_per_shot": he_burn_fine},
        }
    (out / "steel_behind_film.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["scenario,delta_um,He_burn_m2,He_debris_m2,H_isotopes_m2,ion_energy_fraction_into_steel,"
             "dpa_peak_per_shot,dpa_mean_0_5um_per_shot,shots_to_1e18,shots_to_1e19"]
    for name in SCEN:
        for r in res["scenarios"][name]["rows"]:
            lines.append(",".join(str(v) for v in (name, r["delta_um"], r["He_per_m2_per_shot"]["burn"], r["He_per_m2_per_shot"]["debris"],
                                                   r["H_isotopes_per_m2_per_shot"]["total"], r["ion_energy_fraction_into_steel"],
                                                   r["dpa_per_shot_peak"], r["dpa_per_shot_mean_0_5um"],
                                                   r["shots_to_1e18_He_m2"], r["shots_to_1e19_He_m2"])))
    (out / "steel_behind_film.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"main burn alphas (<= {res['main_burn_alpha_top_energy_MeV']:.2f} MeV) stopped by {d_alpha_main_um:.1f} um of Li")
    for name in SCEN:
        sc = res["scenarios"][name]
        print(f"{name}: 1e6 shots to 1e18 beyond {sc['film_for_1e6_shots_to_1e18_um']} um, to 1e19 beyond "
              f"{sc['film_for_1e6_shots_to_1e19_um']} um; all burn He stopped by {sc['film_stopping_all_burn_He_um']} um")
        for r in sc["rows"]:
            print(f"  {r['delta_um']:8.3f} um  He {r['He_per_m2_per_shot']['burn']:.3e} + {r['He_per_m2_per_shot']['debris']:.3e}  "
                  f"H {r['H_isotopes_per_m2_per_shot']['total']:.3e}  Eion->Fe {r['ion_energy_fraction_into_steel']:.4f}"
                  + (f" (prod {r['production_ion_in_Fe_fraction']:.4f})" if 'production_ion_in_Fe_fraction' in r else "")
                  + f"  dpa peak {r['dpa_per_shot_peak']:.2e} mean5 {r['dpa_per_shot_mean_0_5um']:.2e}"
                  f"  N18 {r['shots_to_1e18_He_m2'] or 0:.3g} N19 {r['shots_to_1e19_He_m2'] or 0:.3g}")


if __name__ == "__main__":
    main()
