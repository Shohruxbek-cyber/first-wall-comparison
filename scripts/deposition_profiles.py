"""Stage 3: energy deposition in W and Li from the threat spectrum.

Static deposition only (no thermal response).  Uses the project baseline:
x-rays = digitized Raffray Fig. 2 (2.1 MJ), XCOM photoelectric+incoherent / Henke < 1 keV;
ions = ARIES NRL tables, production spectra (ife_wall.ions, 41.6 MJ), SRIM baseline depth model.
Energies are for the 154 MJ target; fractions are yield-independent (spectra scale with Y).

Outputs (runs/deposition/):
  deposition_W.png/pdf, deposition_Li.png/pdf  energy density vs depth per component
  fractions.json   fraction of threat (x-ray + ion) energy within 10 nm..10 um; per-species depths
  li_film_absorption.json/png  energy absorbed in a Li film of thickness delta on Fe
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.srim import SrimStopping  # noqa: E402
from ife_wall.source_term import SquarePulse  # noqa: E402
from ife_wall.xray import XraySource  # noqa: E402

RHO = {"W": 19250.0, "Li": 505.0, "Fe": 7800.0}  # kg/m^3 at ~300 K (W: Tolias), ~600 K (Li: Davison), steel (BM2005)
E_X, E_I = 2.1e6, 41.6e6  # J, whole target (Raffray Table 1)
OUT = ROOT / "runs" / "deposition"
OUT.mkdir(parents=True, exist_ok=True)


class Slab:
    """Minimal stand-in for ThermalSolver: cells with areal mass m and layer index."""

    def __init__(self, layers):  # [(element, thickness_m, dx0, growth)]
        m, lay, self.els = [], [], []
        for j, (el, L, dx0, g) in enumerate(layers):
            dx = [];
            x = 0.0
            while x < L * (1 - 1e-12):
                d = min(dx0 * g ** len(dx), L - x)
                dx.append(d); x += d
            m += list(np.array(dx) * RHO[el]); lay += [j] * len(dx); self.els.append(el)
        self.m, self.layer = np.array(m), np.array(lay)
        self.dx_arr = self.m / np.array([RHO[self.els[j]] for j in self.layer])
        self.x_edges = np.r_[0.0, np.cumsum(self.dx_arr)]


def components(slab, stopping):
    """Per-cell deposited energy [J, whole target] for x-rays and each ion (group, species)."""
    xs = XraySource(1.0, SquarePulse(0, 1e-9), slab.els, n_bins=200)
    out = {"x-rays": E_X * xs.fractions(slab)}
    bins = load_ion_bins(True)
    ions = IonSpectrumSource(bins, 154e6, 6.5, slab.els, stopping)
    fr = ions.fractions(slab)  # (n_bins, n_cells)
    for b, row in zip(ions.bins, fr):
        key = f"{b.group} {b.species}"
        out[key] = out.get(key, 0) + b.energy * row
    return out


def summarise(slab, comp):
    total_in = E_X + E_I
    dep = sum(comp.values())
    res = {"total_deposited_frac": float(dep.sum() / total_in)}
    for d in (10e-9, 100e-9, 1e-6, 10e-6, 100e-6):
        c = np.interp(d, slab.x_edges, np.r_[0.0, np.cumsum(dep)])
        res[f"within_{d * 1e9:g}nm"] = float(c / total_in)
    depth = {}
    xc = 0.5 * (slab.x_edges[1:] + slab.x_edges[:-1])
    for k, v in comp.items():
        if v.sum() > 0:
            cum = np.cumsum(v) / v.sum()
            depth[k] = {"mean_um": float(np.sum(v * xc) / v.sum() * 1e6),
                        "x50_um": float(np.interp(0.5, cum, slab.x_edges[1:]) * 1e6),
                        "x90_um": float(np.interp(0.9, cum, slab.x_edges[1:]) * 1e6),
                        "deposited_frac_of_component": float(v.sum() / (E_X if k == "x-rays" else
                                                                          sum(b.energy for b in load_ion_bins(True) if f"{b.group} {b.species}" == k)))}
    res["per_component_depth"] = depth
    return res


def plot(slab, comp, el, fname):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xc = 0.5 * (slab.x_edges[1:] + slab.x_edges[:-1])
    groups = {"x-rays": ["x-rays"],
              "burn ions (p, D, T, He)": [k for k in comp if k.startswith("burn")],
              "debris H isotopes": [k for k in comp if k.startswith("debris") and any(s in k for s in ("Protons", "Deuterons", "Tritons"))],
              "debris He": [k for k in comp if k.startswith("debris") and "He" in k],
              "debris C, Au": [k for k in comp if k.startswith("debris") and any(s in k for s in ("12C", "Au"))]}
    area = 4 * np.pi * 6.5 ** 2
    for lab, keys in groups.items():
        e = sum(comp[k] for k in keys) / area / slab.dx_arr  # J/m^3 at the wall (154 MJ, 6.5 m)
        ax.loglog(xc * 1e6, np.maximum(e, 1e-3), label=lab)
    ax.loglog(xc * 1e6, sum(comp.values()) / area / slab.dx_arr, "k-", lw=2, label="total")
    ax.set(xlim=(1e-3, 1e3), ylim=(1e5, 3e11), xlabel=r"depth ($\mu$m)", ylabel="deposited energy density (J/m$^3$)",
           title=f"Energy deposition in {el}: NRL 154 MJ target at R = 6.5 m (single shot, no vapour shielding)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=150)


if __name__ == "__main__":
    st = SrimStopping("projected")
    results = {}
    for el, L in (("W", 1e-3), ("Li", 10e-3)):
        slab = Slab([(el, L, 0.5e-9, 1.04)])
        comp = components(slab, st)
        results[el] = summarise(slab, comp)
        plot(slab, comp, el, f"deposition_{el}")
    # CSDA sensitivity for the depth fractions
    for el, L in (("W", 1e-3), ("Li", 10e-3)):
        slab = Slab([(el, L, 0.5e-9, 1.04)])
        r = summarise(slab, components(slab, SrimStopping("csda")))
        results[f"{el}_csda"] = {k: v for k, v in r.items() if k != "per_component_depth"}
    (OUT / "fractions.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # Li film on Fe: absorbed fraction vs thickness
    film = []
    for delta in np.geomspace(1e-6, 1e-3, 19):
        slab = Slab([("Li", delta, min(0.5e-9 * 20, delta / 200), 1.04), ("Fe", 2e-3, 1e-9, 1.05)])
        comp = components(slab, st)
        inLi = slab.layer == 0
        row = {"delta_um": delta * 1e6}
        tot_x = comp["x-rays"]
        row["xray_in_Li"] = float(tot_x[inLi].sum() / E_X)
        row["xray_in_Fe"] = float(tot_x[~inLi].sum() / E_X)
        ion = sum(v for k, v in comp.items() if k != "x-rays")
        row["ion_in_Li"] = float(ion[inLi].sum() / E_I)
        row["ion_in_Fe"] = float(ion[~inLi].sum() / E_I)
        row["threat_in_Li"] = float((tot_x[inLi].sum() + ion[inLi].sum()) / (E_X + E_I))
        row["threat_in_Fe"] = float((tot_x[~inLi].sum() + ion[~inLi].sum()) / (E_X + E_I))
        film.append(row)
    (OUT / "li_film_absorption.json").write_text(json.dumps(film, indent=1), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    d = [r["delta_um"] for r in film]
    for k, lab in (("threat_in_Li", "all threat energy (x-ray + ion)"), ("ion_in_Li", "ions"), ("xray_in_Li", "x-rays")):
        ax.semilogx(d, [r[k] for r in film], "o-", ms=3, label=lab)
    ax.set(xlabel=r"Li film thickness $\delta$ ($\mu$m)", ylabel="fraction absorbed in the Li film",
           title="Li film on steel: fraction of incident threat energy absorbed in Li", ylim=(0, 1.02))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"li_film_absorption.{ext}", dpi=150)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_component_depth"} for k, v in results.items()}, indent=1))
    for r in film[::3]:
        print(f"delta {r['delta_um']:8.1f} um: threat in Li {r['threat_in_Li']:.3f} (ions {r['ion_in_Li']:.3f}, x-rays {r['xray_in_Li']:.3f}); into steel {r['threat_in_Fe']:.3f}")
