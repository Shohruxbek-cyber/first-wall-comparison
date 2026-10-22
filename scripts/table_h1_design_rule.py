"""Heat-balance column of Table H.1 and the design-rule cycle counts of Table 4 and Section 4.4.

The two formulas are those used for the paper:
  * the mean W/steel interface temperature at periodic steady state (Section 2.7),
        T_if = T_c + q (1/h + L_s/k_s),
    with T_c = 600 K, q the energy absorbed per shot (the ledger of the stored single shot) times the
    repetition rate, L_s = 1 cm and k_s = 33 W m^-1 K^-1, for h = 5, 10 and 20 kW m^-2 K^-1;
  * the allowed load cycles of the surface-load design rule of Mitteau et al. (2024), N = exp(100/F_HF),
    at the heat-flux factors F_HF printed by Gonzalez-Arrabal et al. (2020), Table I.

Reads runs/w_single/baseline/{Prototype,Demo}.json (stage tungsten_single).
Writes results/revision/table_h1_design_rule/table_h1_design_rule.json.
Usage: python scripts/table_h1_design_rule.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "table_h1_design_rule"
REP = {"Prototype": 1.0, "Demo": 10.0}  # Hz
FHF_GA = {"Experimental": 9.7, "Prototype": 14.4, "Demo": 44.4}  # MJ m^-2 s^-1/2, Gonzalez-Arrabal et al. (2020), Table I
T_COOLANT, L_STEEL, K_STEEL = 600.0, 0.01, 33.0


def steady_interface_mean(ledger_deposited_J_m2, rep_Hz, h):
    q = ledger_deposited_J_m2 * rep_Hz  # W/m^2
    return T_COOLANT + q * (1.0 / h + L_STEEL / K_STEEL)


def mitteau_N(fhf):
    return math.exp(100.0 / fhf)


def main():
    table = {}
    for scen, rep in REP.items():
        w = json.loads((ROOT / "runs" / "w_single" / "baseline" / f"{scen}.json").read_text(encoding="utf-8"))
        dep = w["ledger_J_m2"]["deposited"]
        table[scen] = {"absorbed_per_shot_J_m2": dep, "rep_Hz": rep,
                       "mean_interface_K": {str(int(h)): steady_interface_mean(dep, rep, h) for h in (5e3, 1e4, 2e4)}}
    out = {"note": __doc__.strip().split("\n\n")[0],
           "heat_balance": table,
           "design_rule_cycles": {s: {"F_HF_MJ_m2_s05": f, "N": mitteau_N(f)} for s, f in FHF_GA.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_h1_design_rule.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s, v in table.items():
        print(s, ", ".join(f"h = {h} W/m2K: {t:.0f} K" for h, t in v["mean_interface_K"].items()))
    print("design-rule cycles:", ", ".join(f"{s} {v['N']:.2g}" for s, v in out["design_rule_cycles"].items()))


if __name__ == "__main__":
    main()
