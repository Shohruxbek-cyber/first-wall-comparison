"""Verification test 6: comparison with published experiments (honest agreement report, not a fit).

(a) Thermal desorption of He from W (Kornelsen 1972, as modelled by Faney 2013): a slab holding He_4 V (4 He per
    vacancy), 40 K/s ramp.  The model releases one He per peak, at the peaks of He_4V, He_3V, He_2V, HeV.  Compared with
    (i) the Redhead formula for the model's own activation energies Eb + Em and (ii) Kornelsen's measured peaks
    950, 1120, 1220, 1560 K.  Caveat: for k <= 4 the model's Eb ARE Kornelsen's energies, so this checks the
    implementation (rate law, ramp, retrapping), not the physics; and the vacancy density of the experiment is not
    known (MISSING), so only peak temperatures are compared, not retained fractions.
(b) Pulsed implantation with flash heating (Gilliam et al. 2005, as quoted in Sharafat et al. 2005): packets of
    1e16 He/m^2 per cycle at 850 C, flash to 2000 C, 1000 cycles; reported release about 95 % (single crystal) and
    about 70 % (polycrystal), cycle length ~18 s.  Everything not in the quote is an ASSUMPTION (see below).
"""
import sys
import time

import numpy as np
from scipy.optimize import brentq

from _he_common import ROOT, record, style, VER
from ife_wall.he import constants as C
from ife_wall.he import damage as dm
from ife_wall.he.fast import FastHe
from ife_wall.he.model import N_W
from ife_wall.he.sources import ShotSources, SubBin

KB = 8.617333262e-5


def redhead(E, beta, nu=1e13):
    f = lambda T: E / (KB * T) - (np.log(nu * T / beta) - 3.64)
    return brentq(f, 300.0, 4000.0)


def tds(nT=1):
    """He_4 V slab, 40 K/s ramp, returns (T, release rate) and the peak temperatures."""
    edges = np.linspace(0, 100e-9, 101)
    p = C.baseline()
    fm = FastHe(edges, p)
    beta, T0, T1 = 40.0, 300.0, 1800.0
    tend = (T1 - T0) / beta
    fm.set_cycle([0.0, tend], np.tile(np.array([[T0], [T1]]), (1, fm.n)))
    y = np.zeros(fm.N)
    u = y[6:].reshape(fm.n, fm.S)
    nz = 6                                                  # 0-6 nm
    u[:nz, 3 + 3] = 5.0                                      # HV_4: species index 2 + 4 = 6 -> column 6; 5 appm (retrapping negligible)
    ts = np.linspace(0, tend, 1501)
    rel = [y[0]]
    for a, b in zip(ts[:-1], ts[1:]):
        y = fm.integrate(y, a, b, rtol=1e-8, atol=1e-14)
        rel.append(y[0])
    rel = np.array(rel)
    T = T0 + beta * ts
    rate = np.gradient(rel, ts)
    return T, rate, rel


def peaks(T, rate):
    from scipy.signal import find_peaks
    r = rate / rate.max()
    idx, _ = find_peaks(r, height=0.02, prominence=0.02)
    return T[idx], r[idx]


def part_a(plt):
    T, rate, rel = tds()
    pk, h = peaks(T, rate)
    Eb = C.EB_BASE
    red = [redhead(Eb[k - 1] + C.EM_HE, 40.0) for k in (4, 3, 2, 1)]
    korn = [950.0, 1120.0, 1220.0, 1560.0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(T, rate / rate.max(), "k-", lw=2, label="model (He$_4$V slab, 40 K/s)")
    for t in korn:
        ax.axvline(t, color="C3", ls="--", lw=1)
    ax.axvline(korn[0], color="C3", ls="--", lw=1, label="Kornelsen peaks")
    for t in red:
        ax.axvline(t, color="C0", ls=":", lw=1.5)
    ax.axvline(red[0], color="C0", ls=":", lw=1.5, label="Redhead, $E_b + E_m$")
    ax.set(xlabel="temperature (K)", ylabel="He release rate (normalised)", xlim=(500, 1800))
    ax.legend(fontsize=9)
    fig.savefig(str(VER / "test6a_tds.pdf"))
    return dict(model_peaks_K=[float(x) for x in pk], redhead_for_Eb_plus_Em_K=red, kornelsen_K=korn,
                total_released_fraction=float(rel[-1] / rel[-1]), n_peaks=len(pk),
                max_dev_from_redhead_K=float(max(abs(a - b) for a, b in zip(pk, red))) if len(pk) == 4 else None,
                max_dev_from_kornelsen_K=float(max(abs(a - b) for a, b in zip(pk, korn))) if len(pk) == 4 else None)


# ---------------------------------------------------------------------------------------- (b)
def part_b(cycles=1000, label="polycrystal", rho_d=C.RHO_D, flash_K=2273.15, T_imp=1123.15, t_imp=2.0, t_up=4.0,
           t_cycle=18.0, species="3He", E_MeV=1.3, plot=None):
    """One implant + flash cycle repeated `cycles` times.  ASSUMPTIONS (not in the quoted text): the 1e16 He/m^2
    packet is delivered at constant rate in the first t_imp = 2 s at 850 C; the flash is a linear ramp up to 2000 C
    in t_up = 4 s followed by a linear ramp down to 850 C by t_cycle = 18 s ('flash heating/cooling cycle ~18 s'
    is the only number given); 1.3 MeV 3He from a search-engine summary of the abstract (not verified)."""
    from ife_wall.he.grid import make_grid
    edges = make_grid()
    p = C.baseline(sink_V=C.Z_V * rho_d, sink_I=C.Z_I * rho_d)
    fm = FastHe(edges, p)
    E = E_MeV * 1e3 * 1.602176634e-16
    sb = SubBin("imp", species, E, E, E, 1e16, 1e-7, 1e-6)
    S = ShotSources(None, edges, sub_bins=[sb], n_slices=2)
    he = S.he_prof.sum(0) / N_W * 1e6                  # appm per cycle
    dpa = S.dpa_prof.sum(0) * 1e6                      # appm (Frenkel pairs, NRT) per cycle
    ns = 24
    te = np.linspace(0, t_imp, ns + 1)
    G_He = np.tile(he / t_imp, (ns, 1))
    G_V = np.tile(dpa / t_imp, (ns, 1))
    fm.set_sources(te, G_He, G_V)
    tt = np.array([0.0, t_imp, t_imp + 1e-3, t_imp + t_up, t_cycle])
    TT = np.array([T_imp, T_imp, T_imp, flash_K, T_imp])
    fm.set_cycle(tt, np.tile(TT[:, None], (1, fm.n)))
    y = np.zeros(fm.N)
    depth_mean = float((S.he_profile() * S.x * S.dx).sum() / (S.he_profile() * S.dx).sum())
    out = []
    t0 = time.time()
    for c in range(cycles):
        for a, b in ((0.0, te[-1]), (te[-1], t_imp + 1e-3), (t_imp + 1e-3, t_imp + t_up), (t_imp + t_up, t_cycle)):
            y = fm.integrate(y, a, b, rtol=1e-3, atol=1e-9)
        if (c + 1) % 50 == 0 or c < 5:
            tot = fm.totals(y)
            out.append((c + 1, tot["He_surface"] / tot["He_in"], tot["He_mobile"] + tot["He_trapped"]))
    tot = fm.totals(y)
    print(f"  {label}: {cycles} cycles, release fraction {tot['He_surface'] / tot['He_in']:.3f}, "
          f"retained {tot['He_mobile'] + tot['He_trapped']:.3e}/m2, mean depth {depth_mean * 1e6:.2f} um ({time.time() - t0:.0f}s)", flush=True)
    return dict(label=label, cycles=cycles, release_fraction=tot["He_surface"] / tot["He_in"], retained_m2=tot["He_mobile"] + tot["He_trapped"],
                implanted_m2=tot["He_in"], mean_depth_um=depth_mean * 1e6, peak_He_appm_per_cycle=float(he.max()),
                vacancies_per_He=float((dpa * S.dx).sum() / (he * S.dx).sum()), curve=out)


def main():
    plt = style()
    a = part_a(plt)
    print("(a) TDS:", a, flush=True)
    ok_a = a["n_peaks"] == 4 and a["max_dev_from_redhead_K"] is not None and a["max_dev_from_redhead_K"] < 40
    cyc = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    poly = part_b(cyc, "polycrystal (rho_d = 5e12 m^-2)", rho_d=C.RHO_D)
    single = part_b(cyc, "single crystal (rho_d = 2e10 m^-2)", rho_d=C.RHO_D_LOW)
    noflash = part_b(cyc, "polycrystal, no flash (850 C only)", rho_d=C.RHO_D, flash_K=1123.15)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r, c in ((poly, "C0"), (single, "C1"), (noflash, "C2")):
        x, y1, _ = zip(*r["curve"])
        ax.plot(x, np.array(y1) * 100, "o-", color=c, ms=3, label=r["label"])
    ax.axhline(95, color="C1", ls="--", lw=1); ax.axhline(70, color="C0", ls="--", lw=1)
    ax.text(cyc * 0.55, 96, "reported, single crystal ~95 %", fontsize=9); ax.text(cyc * 0.55, 71, "reported, polycrystal ~70 %", fontsize=9)
    ax.set(xlabel="implant + flash cycles (1e16 He/m$^2$ each)", ylabel="He released through the surface (%)", ylim=(0, 105))
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(str(VER / "test6b_gilliam.pdf"))
    record("test6_benchmarks", ok_a, part_a_TDS=a, part_b_flash_heating=dict(polycrystal=poly, single_crystal=single, no_flash=noflash,
           reported=dict(single_crystal=0.95, polycrystal=0.70, source="Gilliam et al. 2005 as quoted in Sharafat et al. 2005 Sec. 6.3 (1000 x 1e16 = 1e19, printed '1e20')")),
           note="pass flag refers to part (a) (4 peaks within 40 K of Redhead); part (b) is reported, not pass/fail")


if __name__ == "__main__":
    main()
