"""Diagnostic (revision of 20 Oct 2026): energy-weighted mean deposition depth in W of a
monoenergetic 3.81 MeV alpha particle, with the same depth model and slab as
scripts/deposition_profiles.py (SrimStopping("projected"); W slab 1 mm, 0.5 nm first cell, x1.04).
It makes traceable the alpha depth quoted in Appendix D of the paper.

Usage: python scripts/diag_alpha_depth.py [--out=results/revision/alpha_depth]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from deposition_profiles import Slab  # noqa: E402  (importing runs nothing: its main block is guarded)
from ife_wall.ions import MEV  # noqa: E402
from ife_wall.srim import SrimStopping  # noqa: E402

if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), ROOT / "results" / "revision" / "alpha_depth"))
    out.mkdir(parents=True, exist_ok=True)
    st = SrimStopping("projected")
    slab = Slab([("W", 1e-3, 0.5e-9, 1.04)])
    xc = 0.5 * (slab.x_edges[1:] + slab.x_edges[:-1])
    res = {}
    for E_MeV in (3.81, 2.08):
        fr = np.asarray(st.fractions(E_MeV * MEV, "4He", slab.m, slab.layer, slab.els))
        cum = np.cumsum(fr) / fr.sum()
        res[f"{E_MeV} MeV"] = {"mean_um": float(np.sum(fr * xc) / fr.sum() * 1e6),
                               "x50_um": float(np.interp(0.5, cum, slab.x_edges[1:]) * 1e6),
                               "x90_um": float(np.interp(0.9, cum, slab.x_edges[1:]) * 1e6),
                               "deposited_fraction": float(fr.sum())}
    (out / "alpha_depth_W.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
