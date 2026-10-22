"""Helium-exclusion films for the steel behind the lithium film, exactly (revision of 21 Oct 2026).

The helium reaching the steel per shot is a step function of the film thickness: a helium sub-bin of
ife_wall.he.sources.split_bins (24 sub-bins per energy bin) reaches the steel when its projected range in lithium
(srim.SrimStopping, projected mode, as in ife_wall.he.steel) exceeds the areal mass of the film.  Sorting the
sub-bins by range gives the film at which the per-shot fluence first falls to a level exactly, where the fine grid
of scripts/steel_behind_film.py interpolates between grid points.  Levels: 1e12 and 1e13 He m^-2 per shot, i.e.
1e6 shots to 1e18 and 1e19 m^-2 with all helium retained, and one full-power year (3.16e8 shots at 10 Hz for Demo,
3.16e7 at 1 Hz for Prototype) to the same fluences.  Also the film that stops every burn helium ion, and the
sub-bin that sets each film.  Production spectra (ife_wall.ions.DEFAULT_ION_SPECTRA); "sensitivity_spectra" repeats
the films with the earlier normalized spectra and with the tilt-only and stretch-only conserving spectra (revision
of 22 Oct 2026).

Usage: python scripts/diag_steel_criteria.py [--out=results/revision/steel_behind_film]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.he.sources import HE_SPECIES, split_bins  # noqa: E402
from ife_wall.library import lithium  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

FPY_S = 365.25 * 24 * 3600
SENSITIVITY = ("normalized", "conserving_tilt", "conserving_stretch")


def film_for_level(R, n, level, rho):
    """Smallest film (um) at which sum(n[R > sigma]) <= level; R in kg/m^2, sorted descending."""
    cum = np.cumsum(n)
    k = int(np.argmax(cum > level)) if (cum > level).any() else None
    return None if k is None else float(R[k] / rho * 1e6)


def setter(R, n, level, sub):
    """The sub-bin at which the cumulative fluence first exceeds the level (the one that sets the film)."""
    cum = np.cumsum(n)
    if not (cum > level).any():
        return None
    s = sub[int(np.argmax(cum > level))]
    return f"{s.group} {s.species} {s.E / 1.602176634e-13:.2f} MeV"


def criteria(st, rho, spectra=None, with_setter=False):
    res = {}
    for name in ("Experimental", "Prototype", "Demo"):
        sub = [s for s in split_bins(SCENARIOS[name], 24, spectra=spectra) if s.species in HE_SPECIES and s.n > 0]
        R = np.array([float(st.range(s.species, "Li", s.E)) for s in sub])
        n = np.array([s.n for s in sub])
        o = np.argsort(-R)
        R, n, sub = R[o], n[o], [sub[i] for i in o]
        levels = {"1e6_shots_to_1e18": 1e18 / 1e6, "1e6_shots_to_1e19": 1e19 / 1e6}
        if SCENARIOS[name].rep_Hz:
            hz = SCENARIOS[name].rep_Hz
            tag = f"1fpy_{hz:g}Hz"
            levels.update({f"{tag}_to_1e18": 1e18 / (FPY_S * hz), f"{tag}_to_1e19": 1e19 / (FPY_S * hz)})
        burn = np.array([s.group == "burn" for s in sub])
        res[name] = {
            "film_um": {k: film_for_level(R, n, lv, rho) for k, lv in levels.items()},
            "film_stopping_all_He_um": float(R[0] / rho * 1e6),
            "film_stopping_all_burn_He_um": float(R[burn][0] / rho * 1e6),
            "He_per_m2_per_shot": float(n.sum())}
        if with_setter:
            res[name]["set_by"] = {k: setter(R, n, lv, sub) for k, lv in levels.items()}
    return res


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "steel_behind_film"))
    out = out if out.is_absolute() else ROOT / out
    rho = float(lithium().rho(np.array(600.0)))
    st = srim.SrimStopping(mode="projected")
    from ife_wall.ions import DEFAULT_ION_SPECTRA
    res = {"note": __doc__.split("Usage:")[0].strip(), "Li_density_kg_m3": rho, "ion_spectra": DEFAULT_ION_SPECTRA,
           "scenarios": criteria(st, rho, with_setter=True),
           "sensitivity_spectra": {sp: criteria(st, rho, sp) for sp in SENSITIVITY if sp != DEFAULT_ION_SPECTRA}}
    for name, r in res["scenarios"].items():
        print(name, r)
    for sp, rr in res["sensitivity_spectra"].items():
        print(sp, {k: {kk: (None if v is None else round(v, 1)) for kk, v in r["film_um"].items()} for k, r in rr.items()})
    (out / "steel_criteria.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
