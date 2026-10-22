"""Energy-weighted mean depth at which the burn and debris helium deposit their energy in Demo tungsten, and the
energy-weighted mean energy of the burn alphas (revision of 22 Oct 2026; Appendices D and H).

Production deposition (ife_wall.ions.IonSpectrumSource with the projected depth model) of the 4He bins of each
group alone, on the mesh of the Demo tungsten case, weighted with the deposited energy per cell.  For both the
production spectra and the spectra used before 22 Oct 2026 ("normalized").  No solver is run.

Usage: python scripts/diag_he_energy_depth.py [--out=results/revision/he_energy_depth]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.ions import KEV_J, IonSpectrumSource, ion_spectra_tables, load_ion_bins  # noqa: E402
from ife_wall.scenarios import SCENARIOS, Options, w_case  # noqa: E402


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "he_energy_depth"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    sc = SCENARIOS["Demo"]
    res = {"note": __doc__.split("Usage:")[0].strip(), "spectra": {}}
    for sp in ("conserving", "normalized"):
        s = w_case(sc, Options(ion_spectra=sp))
        x = s.x_centers()
        r = {}
        for grp in ("burn", "debris"):
            bins = [b for b in load_ion_bins(spectra=sp) if b.species == "4He" and b.group == grp]
            src = IonSpectrumSource(bins, sc.yield_J, sc.R, ["W", "Fe"], srim.SrimStopping("projected"))
            dep = np.array(src.fluence) @ np.asarray(src.fractions(s))
            r[f"{grp}_4He_energy_weighted_mean_deposition_depth_um"] = float((dep * x).sum() / dep.sum() * 1e6)
        d = ion_spectra_tables(sp)[("burn", "4He")]
        e_bin = d["energy"] / d["number"] / KEV_J
        r["burn_4He_energy_weighted_mean_energy_MeV"] = float((d["energy"] * e_bin).sum() / d["energy"].sum() / 1e3)
        r["burn_4He_number_mean_energy_MeV"] = float(d["energy"].sum() / d["number"].sum() / KEV_J / 1e3)
        res["spectra"][sp] = r
        print(sp, {k: round(v, 3) for k, v in r.items()})
    (out / "he_energy_depth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
