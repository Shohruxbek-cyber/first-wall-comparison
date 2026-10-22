"""Production: Li film thickness scan.

Each shot hits a fresh film at T_li on a steel substrate in equilibrium with it (complete
replacement between shots).  Per film thickness delta:
  * fraction of threat energy absorbed in the Li (static deposition at t = 0: incident
    energy x per-cell fractions summed over Li cells);
  * peak substrate (steel) temperature over the shot, pass/fail vs 925 K (Garoz2016);
  * Li evaporated per shot: kg/m^2, um (at rho(T_li)), kg per shot for the whole chamber
    (x 4 pi R^2), and the average vapour number density if spread over (4/3) pi R^3;
  * burn-through flag (film fully evaporated), and whether the Li surface exceeded the
    1800 K upper end of the Davison vapour-pressure data (extrapolated p_sat).
Flow speed for complete replacement: 2R x rep rate (chamber height = 2R, author choice).

Usage: python scripts/run_li_scan.py <scenario> [T_li=600] [option=value ...] [li_lv=handbook_147kJmol]
Output: runs/li_scan/<tag>/<scenario>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall.library import M_LI  # noqa: E402
from ife_wall.scenarios import SCENARIOS, li_case  # noqa: E402
from run_w_single import parse_overrides  # noqa: E402

NA = 6.02214076e23
SUBSTRATE_LIMIT_K = 925.0
DELTAS = np.geomspace(1e-6, 1e-3, 13)

name = sys.argv[1]
kv = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
T_li = float(kv.pop("T_li", 600.0))
li_lv = kv.pop("li_lv", "davison_slope")
opt = parse_overrides([f"{k}={v}" for k, v in kv.items()])
tag = "_".join([f"T{T_li:.0f}K"] + ([f"lv-{li_lv}"] if li_lv != "davison_slope" else []) + [f"{k}-{v}" for k, v in kv.items()])
scen = SCENARIOS[name]
area, vol = 4 * np.pi * scen.R ** 2, 4 / 3 * np.pi * scen.R ** 3
rows = []
d = ROOT / "runs" / "li_scan" / tag
d.mkdir(parents=True, exist_ok=True)
for delta in DELTAS:
    t0 = time.time()
    s = li_case(scen, delta, T_li, opt, li_lv=li_lv)
    rho_li = float(s.materials[0].rho(np.array(T_li)))
    m_li0 = float(s.m[s.layer == 0].sum())
    # static absorbed fraction (t = 0 mesh)
    E_tot = E_li = 0.0
    for src in s.sources:
        fr = np.asarray(src.fractions(s))
        if fr.ndim == 2:
            e = np.array([F for F in src.fluence])  # per-component fluence, J/m^2
            E_tot += e.sum()
            E_li += float(e @ fr[:, s.layer == 0].sum(axis=1))
        else:
            E_tot += src.fluence
            E_li += src.fluence * float(fr[s.layer == 0].sum())
    pk = {"Ts": 0.0, "Tsub": 0.0}

    def rec(sv):
        T = sv.T
        pk["Ts"] = max(pk["Ts"], sv.surface_temperature(T)) if np.any(sv.layer == 0) else pk["Ts"]
        pk["Tsub"] = max(pk["Tsub"], float(T[sv.layer == 1].max()))

    status = "ok"
    try:
        s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
        s.run(5e-3, dt0=1e-9, dT_target=5.0, record=rec)  # let the substrate peak pass
    except RuntimeError as e:
        status = f"stopped at t={s.t:.3e} s: {e}"
    m_left = float(s.m[s.layer == 0].sum())
    burn = m_left < 0.01 * m_li0
    mv = s.mass_evaporated
    row = {"delta_um": delta * 1e6, "absorbed_frac_in_Li": E_li / E_tot, "incident_J_m2": E_tot,
           "peak_Li_surface_K": pk["Ts"], "Li_surface_above_1800K_extrapolated": pk["Ts"] > 1800,
           "peak_substrate_K": pk["Tsub"], "substrate_pass": pk["Tsub"] < SUBSTRATE_LIMIT_K and status == "ok",
           "vaporized_kg_m2": mv, "vaporized_um": mv / rho_li * 1e6, "vaporized_kg_per_shot_chamber": mv * area,
           "vapour_number_density_m3": mv * area / M_LI * NA / vol,
           "vaporized_energy_J_m2": s.ledger.vaporized, "burn_through": burn, "status": status,
           "energy_rel_residual": float(s.ledger.residual()[1]) if status == "ok" else None,
           "wall_s": time.time() - t0}
    rows.append(row)
    print(f"{name} T_li={T_li:.0f} delta={delta * 1e6:8.2f} um: absorbed {row['absorbed_frac_in_Li']:.3f}, Li surf {pk['Ts']:.0f} K, "
          f"substrate {pk['Tsub']:.0f} K ({'PASS' if row['substrate_pass'] else 'FAIL'}), vap {mv:.3e} kg/m2 = {row['vaporized_um']:.3f} um, "
          f"burn={burn}, {status}", flush=True)
passing = [r for r in rows if r["substrate_pass"] and not r["burn_through"]]
out = {"scenario": name, "T_li_K": T_li, "li_lv": li_lv, "options": {k: v for k, v in vars(opt).items() if k != "extra"},
       "substrate_limit_K": SUBSTRATE_LIMIT_K, "rows": rows,
       "min_passing_delta_um": min((r["delta_um"] for r in passing), default=None),
       "flow_speed_for_full_replacement_m_s": 2 * scen.R * scen.rep_Hz if scen.rep_Hz else None}
(d / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("min passing delta (um):", out["min_passing_delta_um"])
