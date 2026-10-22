"""Mean W/steel interface temperature at periodic steady state for a lower steel conductivity (revision of
21 Oct 2026): the heat balance of Section 2.7, extended to neutron heating, exactly as in
scripts/diag_heat_balance_neutron.py, evaluated at k_s = 33 (the BM2005 value of the baseline) and 29 W/(m K),
for Prototype and Demo, without and with neutron heating of the steel at 0.5 and 1 times the W value.

Usage: python scripts/diag_steel_k.py [--out=results/revision/steel_k29]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diag_heat_balance_neutron import L_S, L_W, t_if  # noqa: E402
from ife_wall.neutron import E_W_REF, R_REF, YIELD_REF  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

K_VALUES = (33.0, 29.0)


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "steel_k29"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {"note": __doc__.split("Usage:")[0].strip()}
    for name in ("Prototype", "Demo"):
        sc = SCENARIOS[name]
        w = json.loads((ROOT / "runs" / "w_single" / "baseline" / f"{name}.json").read_text(encoding="utf-8"))
        q_s = w["ledger_J_m2"]["deposited"] * sc.rep_Hz
        e_w = E_W_REF * sc.yield_J / YIELD_REF * (R_REF / sc.R) ** 2
        rows = {}
        for f in (0.0, 0.5, 1.0):
            q_nw = (e_w * L_W * sc.rep_Hz) if f > 0 else 0.0
            q_ns = f * e_w * L_S * sc.rep_Hz
            key = f"steel factor {f:g}" if f > 0 else "no neutrons"
            rows[key] = {f"mean_interface_K_k{k:g}": t_if(q_s, q_nw, q_ns, sc.h, k_s=k) for k in K_VALUES}
            rows[key]["change_K"] = rows[key]["mean_interface_K_k29"] - rows[key]["mean_interface_K_k33"]
        res[name] = {"q_xray_ion_W_m2": q_s, "cases": rows}
        for c, r in rows.items():
            print(name, c, {k: round(v, 2) for k, v in r.items()})
    (out / "heat_balance_steel_k.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
