"""Bounding estimate of vapour shielding -- kept separate from all main results.

For the Li film (Demo, thick film, T_li = 600 K) the fraction of threat-ion energy that a
Li vapour column could stop, using the same SRIM stopping tables (cold, neutral vapour;
gas-solid and plasma-stopping effects neglected):
  lower bound: vapour produced by the x-rays alone before the first ion arrives;
  upper bound: the full per-shot vaporized mass placed in front of the surface.
For W the vaporized mass is negligible (see runs/w_single), so W shielding is ~0.
A self-consistent answer needs radiation hydrodynamics (HEIGHTS, BUCKY).
Also (21 Oct 2026, Section 2.5 of the article): the share of the x-ray energy that the full per-shot
vapour would absorb (production x-ray source on the same column), and the sound speed of the vapour
at the peak surface temperature (monatomic ideal gas, gamma = 5/3).
Output: runs/li_scan/vapour_shielding_bound_Demo.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.library import M_LI, R_GAS  # noqa: E402
from ife_wall.scenarios import SCENARIOS, Options, li_case  # noqa: E402
from ife_wall.srim import SrimStopping  # noqa: E402


class Column:
    def __init__(self, M, n=400):
        self.m = np.full(n, M / n)
        self.layer = np.zeros(n, int)


st = SrimStopping("projected")
bins = load_ion_bins(True)
s = li_case(SCENARIOS["Demo"], 1e-3, 600.0, Options())
xs, ions = s.sources
s.sources = [xs]
s._frac_cache = None
s.run(ions.t_first, dt0=1e-13, dT_target=5.0)
M_x = s.mass_evaporated
full = ROOT / "runs" / "li_scan" / "T600K" / "Demo.json"
M_full = float(np.median([r["vaporized_kg_m2"] for r in json.loads(full.read_text(encoding="utf-8"))["rows"]
                          if r["delta_um"] >= 100]))
res = {"assumptions": "cold neutral Li vapour, SRIM Li stopping per unit mass, normal incidence"}
for label, M in (("lower_xray_vapour_only", M_x), ("upper_full_per_shot_vapour", M_full)):
    src = IonSpectrumSource(bins, 154e6, 6.5, ["Li"], st)
    fr = src.fractions(Column(max(M, 1e-30)))
    E = np.array([b.energy for b in src.bins])
    res[label] = {"column_kg_m2": M, "ion_energy_fraction_stopped": float(E @ fr.sum(axis=1) / E.sum())}
    print(label, res[label])
T_s = float(np.median([r["peak_Li_surface_K"] for r in json.loads(full.read_text(encoding="utf-8"))["rows"]
                       if r["delta_um"] >= 100]))
res["xray_energy_fraction_absorbed_by_full_per_shot_vapour"] = float(xs.fractions(Column(M_full)).sum())
res["vapour_sound_speed_m_s"] = {"T_surface_K": T_s, "gamma": 5 / 3, "value": float(np.sqrt(5 / 3 * R_GAS * T_s / M_LI))}
print("x-ray share absorbed by the full per-shot vapour", res["xray_energy_fraction_absorbed_by_full_per_shot_vapour"],
      "| sound speed", res["vapour_sound_speed_m_s"])
(ROOT / "runs" / "li_scan" / "vapour_shielding_bound_Demo.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
