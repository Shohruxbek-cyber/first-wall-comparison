"""Neutron displacement rate in tungsten (revision of 21 Oct 2026): the HAPL figure used in the
helium model (Sharafat et al. 2005, Sec. 3.2: of the order of 1e-6 dpa/s at 5 Hz for the 154 MJ target at 6.5 m,
material not stated; 2e-7 dpa per shot, Table I.3) against the tungsten-specific values of Gilbert et al. (2013),
4.5 dpa/fpy (EFF-1.1, NJOY) and 14.5 dpa/fpy (FISPACT-II, TENDL-2011) in the first-wall armour of the 3 GWth CCFE
2009 magnetic-fusion DEMO model (Section 2.3 of that paper; its neutron wall load is not given there, so the
values are converted per full-power year, not scaled), and the time for the neutrons alone to bring tungsten to
0.1 dpa at 10 Hz (Demo).  Also the neutron wall load of each scenario (109 MJ of neutrons per 154 MJ shot,
Raffray 2005, scaled by yield, as in scripts/diag_power_density.py) and the HAPL per-shot rate scaled with the
neutron energy per unit wall area, which the helium model does not do (it uses 2e-7 dpa per shot everywhere).

Usage: python scripts/diag_neutron_dpa.py [--out=results/revision/neutron_dpa]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, YIELD_REF_J  # noqa: E402

FPY_S = 365.25 * 24 * 3600.0
HAPL_DPA_S, HAPL_HZ = 1e-6, 5.0        # Sharafat et al. 2005, Sec. 3.2
HAPL_DPA_SHOT = HAPL_DPA_S / HAPL_HZ    # 2e-7, Table I.3
GILBERT_W_DPA_FPY = {"EFF-1.1": 4.5, "TENDL-2011": 14.5}
NEUTRON_REF_J, R_REF = 109e6, 6.5
DOSE = 0.1


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "neutron_dpa"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    demo = SCENARIOS["Demo"]
    res = {"note": __doc__.split("Usage:")[0].strip(), "HAPL_dpa_per_shot": HAPL_DPA_SHOT,
           "HAPL_dpa_per_fpy_at_10Hz": HAPL_DPA_SHOT * demo.rep_Hz * FPY_S,
           "HAPL_shots_to_0.1dpa": DOSE / HAPL_DPA_SHOT, "HAPL_hours_to_0.1dpa_at_10Hz": DOSE / HAPL_DPA_SHOT / demo.rep_Hz / 3600,
           "Gilbert2013_W": {}, "scenarios": {}}
    for lib, r in GILBERT_W_DPA_FPY.items():
        per_shot = r / (FPY_S * demo.rep_Hz)
        res["Gilbert2013_W"][lib] = {"dpa_per_fpy": r, "dpa_per_Demo_shot_at_10Hz": per_shot,
                                     "ratio_HAPL_to_this": HAPL_DPA_SHOT / per_shot,
                                     "days_to_0.1dpa_at_10Hz": DOSE / per_shot / demo.rep_Hz / 86400}
    for name, sc in SCENARIOS.items():
        e_n = NEUTRON_REF_J * sc.yield_J / YIELD_REF_J / (4 * math.pi * sc.R ** 2)
        res["scenarios"][name] = {"neutron_energy_J_m2_per_shot": e_n,
                                  "neutron_wall_load_MW_m2": (sc.rep_Hz or 1.0) * e_n / 1e6,
                                  "HAPL_rate_scaled_per_shot": HAPL_DPA_SHOT * sc.yield_J / YIELD_REF_J * (R_REF / sc.R) ** 2}
    (out / "neutron_dpa.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
