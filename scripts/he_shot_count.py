"""He fluence per shot and shots to the pulsed damage threshold, with the He
retention fraction from TRIM backscatter (W), weighted over the He spectrum.
Output: runs/source_term/he_shot_count.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import DEFAULT_ION_SPECTRA, load_ion_bins  # noqa: E402
from ife_wall.srim import SrimStopping  # noqa: E402

st = SrimStopping()
out = {}
for spectra in (DEFAULT_ION_SPECTRA, "raw"):  # labels: the production construction ("normalized" before 22 Oct 2026)
    bins = load_ion_bins(spectra=spectra)
    for scen, (Y, R) in {"Experimental": (20e6, 5.0), "Prototype": (50e6, 6.5), "Demo": (154e6, 6.5)}.items():
        sc = Y / 154e6 / (4 * np.pi * R ** 2)
        res = {}
        for label, sel in (("burn 4He", lambda b: b.group == "burn" and b.species == "4He"),
                           ("burn He", lambda b: b.group == "burn" and b.species in ("4He", "3He")),
                           ("burn+debris 4He", lambda b: b.species == "4He"),
                           ("all He", lambda b: b.species in ("4He", "3He"))):
            hb = [b for b in bins if sel(b)]
            n_inc = sum(b.number for b in hb) * sc
            n_ret = sum(b.number * (1 - st.reflected_number_fraction(b.species, "W", b.E_mean)) for b in hb) * sc
            res[label] = dict(incident_per_m2=n_inc, retained_per_m2=n_ret, retained_frac=n_ret / n_inc,
                              shots_1e18=1e18 / n_ret, shots_1e19=1e19 / n_ret)
        out[f"{scen} ({spectra})"] = res
(ROOT / "runs" / "source_term").mkdir(parents=True, exist_ok=True)
(ROOT / "runs" / "source_term" / "he_shot_count.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    print(k, {lab: round(r["shots_1e19"]) for lab, r in v.items()})
