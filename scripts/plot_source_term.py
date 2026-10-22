"""Stage 2 diagnostic: threat ion power at the wall vs time (HiPER scenarios) and fluence checks."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import DEFAULT_ION_SPECTRA, ION_MASS_U, IonSpectrumSource, load_ion_bins  # noqa: E402

SCEN = {"Experimental": (20e6, 5.0), "Prototype": (50e6, 6.5), "Demo": (154e6, 6.5)}
XRAY_MJ_REF, ION_MJ_REF = 2.1, 41.6  # Raffray2005 Table 1 (19.5 + 22.1)

bins = load_ion_bins(normalize=True)
out = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
t = np.logspace(-8, -4, 1500)
for name, (Y, R) in SCEN.items():
    src = IonSpectrumSource(bins, Y, R, ["W"], stopping=None)
    E_cum = np.array([src.energy(0.0, tt) for tt in t])
    P = np.gradient(E_cum, t)
    ax[0].loglog(t * 1e6, np.maximum(P, 1e-3), label=f"{name}: {Y/1e6:.0f} MJ, R = {R} m")
    area = 4 * np.pi * R ** 2
    he = sum(b.number for b in bins if b.species == "4He") * Y / 154e6 / area
    he_burn = sum(b.number for b in bins if b.species == "4He" and b.group == "burn") * Y / 154e6 / area
    t50 = t[np.searchsorted(E_cum, 0.5 * E_cum[-1])]
    out[name] = {"ion_fluence_J_m2": src.total_fluence(), "xray_fluence_J_m2": XRAY_MJ_REF * 1e6 * Y / 154e6 / area,
                 "GA2020_TableI_ion_fluence_J_m2": {"Experimental": 1.7e4, "Prototype": 2.5e4, "Demo": 7.7e4}[name],
                 "He4_per_m2_per_shot": he, "He4_burn_only_per_m2_per_shot": he_burn,
                 "shots_to_1e19_He": 1e19 / he, "shots_to_1e18_He": 1e18 / he,
                 "t_half_ion_energy_us": t50 * 1e6,
                 "first_arrival_us": src.t_first * 1e6, "last_arrival_us": src.t_last * 1e6}  # the solver's on/off events
    # per-group arrival windows
    for g in ("burn", "debris"):
        sb = [b for b in bins if b.group == g]
        s_g = IonSpectrumSource(sb, Y, R, ["W"], stopping=None)
        Eg = np.array([s_g.energy(0.0, tt) for tt in t])
        out[name][f"{g}_t10_t50_t90_us"] = [float(t[np.searchsorted(Eg, f * Eg[-1])] * 1e6) for f in (0.1, 0.5, 0.9)]
ax[0].set(xlabel=r"time after ignition ($\mu$s)", ylabel="ion power at wall (W/m$^2$)", ylim=(1e6, 1e12),
          title="(a) ion power at the first wall (time of flight)")
ax[0].legend(fontsize=9)

# (b) per-species contribution, Demo
Y, R = SCEN["Demo"]
for (g, s), ls in zip([(g, s) for g in ("burn", "debris") for s in ("Deuterons", "Tritons", "4He", "Protons", "12C", "Au")],
                      ["-"] * 6 + ["--"] * 6):
    sb = [b for b in bins if b.group == g and b.species == s]
    if not sb:
        continue
    src = IonSpectrumSource(sb, Y, R, ["W"], stopping=None)
    E_cum = np.array([src.energy(0.0, tt) for tt in t])
    ax[1].loglog(t * 1e6, np.maximum(np.gradient(E_cum, t), 1e-3), ls, label=f"{g} {s}")
ax[1].set(xlabel=r"time after ignition ($\mu$s)", ylabel="W/m$^2$", ylim=(1e6, 1e12),
          title="(b) Demo, by species (solid: burn, dashed: debris)")
ax[1].legend(fontsize=8, ncol=2)
fig.suptitle(f"Ion spectra: ARIES NRL 154 MJ tables ({DEFAULT_ION_SPECTRA} bins); 20/50 MJ scaled by yield (ESTIMATED)", y=1.02)
fig.tight_layout()
d = ROOT / "runs" / "source_term"
d.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(d / f"ion_power_vs_time.{ext}", dpi=150, bbox_inches="tight")
(d / "source_term_checks.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
