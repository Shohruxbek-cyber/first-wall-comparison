"""Ion penetration in lithium against Hassanein and Morozov (2002), Fig. 8 (revision of 20 Oct 2026,
benchmark papers): "Ranges and stopping power in 1 mm Li over 1 cm CFC composite wall", HEIGHTS-IFE,
NRL direct-drive target ions.  Their film sits on carbon composite, not steel; the comparison uses
only the part of the curves inside the lithium.

Reference values were read by eye from the figure of the ISFNT-6 preprint (page 16), with a reading
uncertainty of about 0.5 um for the debris panel and 0.005 cm / 0.3 MeV for the fast-ion panel:
the starting energy of each curve and its end point (debris ions, all stopping in the lithium), or
the energy left after 1 mm of lithium (fast H, D, T), or the end point (fast 4He).  Their lithium
density is not stated, so depths of this work are given for the SRIM solid density (534 kg/m^3)
and for liquid lithium at 600 K (502 kg/m^3, the film of this work).  This work: SRIM-2012 tables
for ions in Li, continuous-slowing-down (CSDA) path and the projected range of the production depth
model (f_TRIM = 1 in lithium).

Usage: python scripts/benchmark_hm2002_li.py [--out=results/benchmarks/hm2002]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402

MEV = 1.602176634e-13
RHO = {"SRIM solid, 534 kg/m3": 534.0, "liquid at 600 K, 502 kg/m3": 562.0 - 0.100 * 600.0}
DEBRIS = {"12C": (3.42, 13.3), "4He": (1.10, 14.5), "3He": (0.83, 10.0), "Tritons": (0.83, 28.5),
          "Deuterons": (0.55, 19.2), "Protons": (0.27, 9.5)}  # MeV at the surface; end point in um
FAST_EXIT = {"Protons": (23.1, 22.3), "Deuterons": (16.3, 14.0), "Tritons": (16.7, 13.4)}  # MeV in, MeV after 1 mm
FAST_RANGE = {"4He": (18.4, 750.0)}  # MeV; end point in um


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "benchmarks" / "hm2002"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    csda, proj = srim.SrimStopping("csda"), srim.SrimStopping("projected")
    res = {"note": __doc__.split("Usage:")[0].strip(), "debris_end_point_um": {}, "fast_energy_after_1mm_MeV": {},
           "fast_end_point_um": {}}
    for sp, (E0, ref) in DEBRIS.items():
        E = E0 * MEV
        res["debris_end_point_um"][sp] = {
            "E0_MeV": E0, "HM2002_um": ref,
            **{f"csda, {k}": float(csda.range(sp, "Li", E)) / rho * 1e6 for k, rho in RHO.items()},
            **{f"projected, {k}": float(proj.range(sp, "Li", E)) / rho * 1e6 for k, rho in RHO.items()}}
    for sp, (E0, ref) in FAST_EXIT.items():
        E = E0 * MEV
        row = {"E0_MeV": E0, "HM2002_MeV": ref}
        E_max = float(csda.table(sp, "Li").E[-1])
        if E > E_max:  # the range table would be clamped at its last row: no comparison
            row["not_compared"] = f"above the {E_max / MEV:g} MeV end of the SRIM table"
            res["fast_energy_after_1mm_MeV"][sp] = row
            continue
        for k, rho in RHO.items():
            mu = 1e-3 * rho  # areal mass of 1 mm
            row[f"csda, {k}"] = float(csda.energy_at_range(sp, "Li", np.array(float(csda.range(sp, "Li", E)) - mu))) / MEV
        res["fast_energy_after_1mm_MeV"][sp] = row
    for sp, (E0, ref) in FAST_RANGE.items():
        E = E0 * MEV
        res["fast_end_point_um"][sp] = {
            "E0_MeV": E0, "HM2002_um": ref,
            **{f"csda, {k}": float(csda.range(sp, "Li", E)) / rho * 1e6 for k, rho in RHO.items()},
            **{f"projected, {k}": float(proj.range(sp, "Li", E)) / rho * 1e6 for k, rho in RHO.items()}}
    (out / "hm2002_li_ranges.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for sec in ("debris_end_point_um", "fast_energy_after_1mm_MeV", "fast_end_point_um"):
        for sp, r in res[sec].items():
            print(sec, sp, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
