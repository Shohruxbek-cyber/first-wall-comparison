"""Vapor shielding with larger stopping factors (revision of 20 Oct 2026; s = 1-2 is an
assumed range): the thick Demo film (1 mm, T_Li = 600 K, single shot to 5 ms) with
(f_rr, s) = (0, 5), (1, 5), (0, 10), (1, 10), compared with the unshielded production row.

Usage: python scripts/diag_vapor_stopping.py [--out=results/vapor_shielding/stopping_factor]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_vapor_shielding import li_shot  # noqa: E402  (same procedure as the production scan)

if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "vapor_shielding" / "stopping_factor"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / "runs/li_scan/T600K/Demo.json").read_text(encoding="utf-8"))["rows"][12]
    rows = []
    for case in ((0.0, 5.0, 0.0), (1.0, 5.0, 0.0), (0.0, 10.0, 0.0), (1.0, 10.0, 0.0)):
        r = li_shot(("Demo", 12, case))
        r["ratio_to_unshielded"] = r["vaporized_kg_m2"] / base["vaporized_kg_m2"]
        rows.append(r)
        print(r["case"], r["vaporized_kg_m2"], r["ratio_to_unshielded"], flush=True)
    (out / "stopping_factor_Demo_1mm.json").write_text(json.dumps({"unshielded_kg_m2": base["vaporized_kg_m2"], "rows": rows},
                                                                   indent=1), encoding="utf-8")
