"""Mean W/steel interface temperature at periodic steady state with neutron heating (revision of
20 Oct 2026), from the 1D heat balance of Section 2.7 extended to volumetric heating:

    T_if = T_c + (q_s + q_nW) (1/h + L_s/k_s) + q_nS (1/h + L_s/(2 k_s)),

where q_s is the repetition rate times the energy absorbed per shot from the x-rays and ions (the
ledger of the stored single shot), q_nW the neutron heat of the 1 mm of tungsten, which crosses
the whole steel, and q_nS the neutron heat of the steel, deposited uniformly in it (exact in this
model: constant k_s, linear boundary condition, no net storage per cycle).  Neutron heating of
tungsten from ife_wall.neutron (Sharafat et al. 2005, scaled by yield and 1/R^2); steel factor
0.5 and 1 (ASSUMPTION), and the case without neutrons for reference.  Also gives the coolant
heat-transfer coefficient and the steel conductivity at which the mean interface is at 925 K.

Usage: python scripts/diag_heat_balance_neutron.py [--out=results/neutron/heat_balance]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.neutron import E_W_REF, R_REF, YIELD_REF  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

T_C, L_W, L_S, K_S, LIMIT = 600.0, 1e-3, 1e-2, 33.0, 925.0
L_S_THIN = 2.5e-3  # steel thickness that puts the coolant 3.5 mm behind the surface


def t_if(q_s, q_nw, q_ns, h, k_s=K_S, L_s=L_S):
    return T_C + (q_s + q_nw) * (1.0 / h + L_s / k_s) + q_ns * (1.0 / h + L_s / (2.0 * k_s))


def solve(f, lo, hi, target=LIMIT):
    """Bisection for f(x) = target on [lo, hi] (f monotone)."""
    flo = f(lo) - target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) - target) * flo > 0:
            lo, flo = mid, f(mid) - target
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "neutron" / "heat_balance"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {"note": __doc__.split("Usage:")[0].strip()}
    for name in ("Prototype", "Demo"):
        sc = SCENARIOS[name]
        w = json.loads((ROOT / "runs" / "w_single" / "baseline" / f"{name}.json").read_text(encoding="utf-8"))
        q_s = w["ledger_J_m2"]["deposited"] * sc.rep_Hz
        e_w = E_W_REF * sc.yield_J / YIELD_REF * (R_REF / sc.R) ** 2  # J/m^3 per shot in tungsten
        rows = {}
        for f in (0.0, 0.5, 1.0):
            q_nw = (e_w * L_W * sc.rep_Hz) if f > 0 else 0.0
            q_ns = f * e_w * L_S * sc.rep_Hz
            h0 = sc.h
            rows[f"steel factor {f:g}" if f > 0 else "no neutrons"] = {
                "q_xray_ion_W_m2": q_s, "q_neutron_W_W_m2": q_nw, "q_neutron_steel_W_m2": q_ns,
                "mean_interface_K": t_if(q_s, q_nw, q_ns, h0),
                "h_for_925K_W_m2K": solve(lambda h: t_if(q_s, q_nw, q_ns, h), 1e3, 1e6) if t_if(q_s, q_nw, q_ns, h0) > LIMIT else None,
                "steel_k_for_925K_W_mK": (solve(lambda k: t_if(q_s, q_nw, q_ns, h0, k), K_S, 1e4)
                                          if t_if(q_s, q_nw, q_ns, h0) > LIMIT and t_if(q_s, q_nw, q_ns, h0, 1e4) < LIMIT else None)}
        thin = {}
        for f in (0.0, 0.5, 1.0):  # coolant 3.5 mm behind the surface: 2.5 mm of steel, its neutron heat scaled with it
            q_nw = (e_w * L_W * sc.rep_Hz) if f > 0 else 0.0
            q_ns = f * e_w * L_S_THIN * sc.rep_Hz
            thin[f"steel factor {f:g}" if f > 0 else "no neutrons"] = {
                "q_neutron_steel_W_m2": q_ns, "mean_interface_K": t_if(q_s, q_nw, q_ns, sc.h, L_s=L_S_THIN)}
        res[name] = {"h_W_m2K": sc.h, "neutron_heating_W_J_m3_per_shot": e_w, "cases": rows,
                     "steel_2.5mm_cases": thin}
    (out / "heat_balance.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name in ("Prototype", "Demo"):
        for c, r in res[name]["cases"].items():
            print(name, c, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
