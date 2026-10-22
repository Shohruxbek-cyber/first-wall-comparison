"""Independent checks of the SRIM data (data/srim/) and of our depth-profile model.

1. Parse all 21 SR Module tables; check energy coverage.
2. SRIM vs NIST PSTAR/ASTAR for p and He-4 in W (S_e, S_n, projected range).
3. Equal-velocity check: S_e(d, E) and S_e(t, E) vs S_e(p, E m_p/m).
4. For each TRIM run: recompute from the raw IONIZ + PHONON files the 50 % and 90 %
   deposited-energy depths and the energy closure; compare with summary.csv.
5. Our depth model (SrimStopping, projected and CSDA modes) vs the TRIM profiles:
   50 % and 90 % energy depths and the deposited-energy fraction.
Outputs: runs/validation/srim_checks.json, runs/validation/srim_profiles.png
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import StoppingTable, csda_fractions_layered  # noqa: E402
from ife_wall.srim import (DATA, EV, MASS_U, SPECIES_CODE, SrimStopping, SrimTable, parse_sr_table,  # noqa: E402
                           trim_profile)

out = {}
CODE2SP = {v: k for k, v in SPECIES_CODE.items()}

# 1. tables
cov = {}
for ion in ("p", "d", "t", "he3", "he4", "c12", "au"):
    for tgt in ("W", "Li", "Fe"):
        d = parse_sr_table(DATA / f"{ion}_in_{tgt}.txt")
        cov[f"{ion}_in_{tgt}"] = dict(rows=len(d["E"]), E_min_keV=d["E"][0] / EV / 1e3, E_max_MeV=d["E"][-1] / EV / 1e6,
                                      rho_g_cm3=d["rho"] / 1e3)
out["tables"] = cov

# 2. SRIM vs NIST (W)
cmp = []
for ion, prog in (("p", "PSTAR"), ("he4", "ASTAR")):
    s = SrimTable(ion, "W")
    n = np.genfromtxt(ROOT / "data" / f"{prog.lower()}_074.csv", delimiter=",", skip_header=2)
    for E_MeV in (0.01, 0.1, 1.0, 3.5, 10.0):
        E = E_MeV * 1e6 * EV
        nSe = np.interp(E_MeV, n[:, 0], n[:, 1]) * 1e6 * EV * 0.1  # MeV cm2/g -> J m2/kg
        nSn = np.interp(E_MeV, n[:, 0], n[:, 2]) * 1e6 * EV * 0.1
        nRp = np.interp(E_MeV, n[:, 0], n[:, 5]) * 10.0  # g/cm2 -> kg/m2
        cmp.append(dict(ion=ion, E_MeV=E_MeV, Se_ratio=float(np.interp(E, s.E, s.Se) / nSe),
                        Sn_ratio=float(np.interp(E, s.E, s.Sn) / nSn), Rp_ratio=float(s.proj_range(E) / nRp)))
out["srim_vs_nist_W"] = cmp

# 3. equal velocity
ev = []
for tgt in ("W", "Li", "Fe"):
    p = SrimTable("p", tgt)
    for ion in ("d", "t"):
        s = SrimTable(ion, tgt)
        E = np.geomspace(s.E[0] * 3, s.E[-1] / 3, 50)
        r = np.interp(E, s.E, s.Se) / np.interp(E * MASS_U["p"] / MASS_U[ion], p.E, p.Se)
        ev.append(dict(ion=ion, target=tgt, max_dev=float(np.abs(r - 1).max())))
out["equal_velocity"] = ev


# 4 + 5. TRIM runs
def depth_at(x, q, w, frac):
    c = np.cumsum(q * w)
    return float(np.interp(frac * c[-1], c, x + w / 2))


summ = []
for f in (DATA / "trim" / "summary.csv", DATA / "trim_supplementary" / "summary.csv"):
    summ += list(csv.DictReader(open(f, newline="")))
runs = []
fig, axs = plt.subplots(3, 5, figsize=(20, 11))
for ax, r in zip(axs.flat, summ):
    run_dir = (DATA / "trim" / r["run"]) if (DATA / "trim" / r["run"]).exists() else (DATA / "trim_supplementary" / r["run"])
    x, q, w = trim_profile(run_dir)
    E0 = float(r["E_keV"]) * 1e3 * EV
    dep = float(np.sum(q * w))
    x50, x90 = depth_at(x, q, w, 0.5), depth_at(x, q, w, 0.9)
    row = dict(run=r["run"], ion=r["ion"], target=r["target"], E_keV=float(r["E_keV"]),
               trim_x50_um=x50 * 1e6, trim_x90_um=x90 * 1e6,
               summary_x50_um=float(r["half_energy_depth_um"]), summary_x90_um=float(r["energy90_depth_um"]),
               trim_deposited_frac=dep / E0, trim_backscatter_E_frac=float(r["backscattered_energy_fraction"]),
               trim_closure_recomputed=dep / E0 + float(r["backscattered_energy_fraction"]),
               summary_closure=float(r["energy_closure"]))
    rho = float(r["density_g_cm3"]) * 1e3
    thick = float(r["thickness_um"]) * 1e-6
    el = r["target"]
    ax.plot(x * 1e6, q / E0 * 1e-6, "k-", lw=1.5, label="TRIM")
    for mode, ls in (("projected", "C0-"), ("csda", "C3--")):
        st = SrimStopping(mode=mode)
        nc = 2000
        m = np.full(nc, thick * rho / nc)
        fr = csda_fractions_layered(E0, CODE2SP[r["ion"]], m, np.zeros(nc, int), [el], st)
        xc = (np.arange(nc) + 0.5) * thick / nc
        wc = np.full(nc, thick / nc)
        row[f"{mode}_x50_um"] = depth_at(xc, fr, wc / wc, 0.5) * 1e6 if fr.sum() > 0 else np.nan
        # depth_at with unit weights: fr already holds per-cell fractions
        c = np.cumsum(fr)
        row[f"{mode}_x50_um"] = float(np.interp(0.5 * c[-1], c, xc + wc / 2)) * 1e6
        row[f"{mode}_x90_um"] = float(np.interp(0.9 * c[-1], c, xc + wc / 2)) * 1e6
        row[f"{mode}_deposited_frac"] = float(fr.sum())
        row[f"{mode}_x50_over_trim"] = row[f"{mode}_x50_um"] / (x50 * 1e6)
        row[f"{mode}_x90_over_trim"] = row[f"{mode}_x90_um"] / (x90 * 1e6)
        ax.plot(xc * 1e6, fr / wc / 1e6, ls, lw=1, label=f"model ({mode})")
    ax.set_title(f"{r['run']}: {r['ion']} {float(r['E_keV']):g} keV -> {el}", fontsize=10)
    ax.set_xlim(0, min(thick, 2.2 * x90) * 1e6)
    ax.set_xlabel(r"depth ($\mu$m)", fontsize=9)
    ax.set_ylabel(r"dE/dx / E$_0$ (1/$\mu$m)", fontsize=9)
    runs.append(row)
axs.flat[0].legend(fontsize=8)
for ax in axs.flat[len(summ):]:
    ax.axis("off")
fig.suptitle("Deposited-energy depth profiles: TRIM (SRIM-2012.03) vs this model (SRIM tables + TRIM range factor + TRIM reflection)")
fig.tight_layout()
(ROOT / "runs" / "validation").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "runs" / "validation" / "srim_profiles.png", dpi=120)
fig.savefig(ROOT / "runs" / "validation" / "srim_profiles.pdf")
out["trim_runs"] = runs
(ROOT / "runs" / "validation" / "srim_checks.json").write_text(json.dumps(out, indent=1))

print("SRIM vs NIST (W):")
for c in cmp:
    print(f"  {c['ion']:3s} {c['E_MeV']:5g} MeV  Se {c['Se_ratio']:.3f}  Sn {c['Sn_ratio']:.3f}  Rp {c['Rp_ratio']:.3f}")
print("equal-velocity max deviation:", max(e["max_dev"] for e in ev))
print("tables:", len(cov), "; min rows", min(v["rows"] for v in cov.values()))
print(f"{'run':15s} {'trim x50':>9s} {'summ':>8s} {'proj/TRIM x50':>13s} {'x90':>6s} {'csda/TRIM x50':>13s} {'x90':>6s} {'dep TRIM':>8s} {'dep proj':>8s} {'closure':>7s}")
for r in runs:
    print(f"{r['run']:15s} {r['trim_x50_um']:9.4g} {r['summary_x50_um']:8.4g} {r['projected_x50_over_trim']:13.3f} {r['projected_x90_over_trim']:6.3f} "
          f"{r['csda_x50_over_trim']:13.3f} {r['csda_x90_over_trim']:6.3f} {r['trim_deposited_frac']:8.3f} {r['projected_deposited_frac']:8.3f} {r['trim_closure_recomputed']:7.3f}")
