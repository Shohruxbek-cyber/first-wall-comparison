"""Diagnostic figure for the verification stage: numerical vs analytical solutions.

Also writes the plotted curves to runs/verification/verification_curves.npz (added 21 Oct 2026),
from which scripts/make_article_figures.py draws Fig. C.1; the figure itself is unchanged."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from ife_wall.materials import constant_material  # noqa: E402
from ife_wall.source_term import BeerLawSource, SquarePulse  # noqa: E402
from ife_wall.thermal import FrontBC, Layer, Mesh, ThermalSolver  # noqa: E402
from test_verification import ALPHA, CP, K, RHO, beer_exact  # noqa: E402

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) constant flux: surface temperature history
mat = constant_material("test", RHO, K, CP)
q, t_end = 1e9, 1e-5
L = 20 * np.sqrt(ALPHA * t_end)
s = ThermalSolver(Mesh.build([Layer(mat, L, L / 2e4, 1.1)], 300.0), 300.0, front=FrontBC("flux", flux=lambda t: q))
hist = []
s.run(t_end, dt0=t_end * 1e-6, dT_target=1.0, record=lambda sv: hist.append((sv.t, sv.surface_temperature())))
t, Ts = np.array(hist).T
ax[0].loglog(t, Ts - 300, "o", ms=3, mfc="none", label="solver")
tt = np.logspace(np.log10(t[0]), np.log10(t_end), 200)
ax[0].loglog(tt, 2 * q / K * np.sqrt(ALPHA * tt / np.pi), "k-", lw=1, label=r"$2q/k\,\sqrt{\alpha t/\pi}$")
curves = {"a_t_s": t, "a_dT_K": Ts - 300, "a_t_exact_s": tt, "a_dT_exact_K": 2 * q / K * np.sqrt(ALPHA * tt / np.pi)}
ax[0].set(xlabel="time (s)", ylabel=r"$T_s - T_0$ (K)", title="(a) constant surface flux, 1 GW/m$^2$")
ax[0].legend()

# (b) Beer-law instantaneous source: profiles
F, mu, tau = 100.0, 1 / 0.5e-6, 1e-13
s = ThermalSolver(Mesh.build([Layer(mat, 200e-6, 2e-9, 1.05)], 300.0), 300.0,
                  sources=[BeerLawSource(F, SquarePulse(0.0, tau), [mu / RHO])])
curves["b_times_s"] = np.array((1e-10, 1e-9, 1e-8, 1e-7))
for i, (tk, c) in enumerate(zip((1e-10, 1e-9, 1e-8, 1e-7), ("C0", "C1", "C2", "C3"))):
    s.run(tk, dt0=1e-15, dT_target=0.5)
    x = s.x_centers()
    ax[1].semilogx(x * 1e6, s.T - 300, "o", ms=3, mfc="none", color=c)
    ax[1].semilogx(x * 1e6, beer_exact(x, tk - tau / 2, F, mu), "-", color=c, lw=1, label=f"t = {tk:.0e} s")
    curves[f"b_x_um_{i}"] = x * 1e6
    curves[f"b_dT_K_{i}"] = s.T - 300
    curves[f"b_dT_exact_K_{i}"] = beer_exact(x, tk - tau / 2, F, mu)
ax[1].set(xlim=(1e-3, 30), xlabel=r"depth ($\mu$m)", ylabel=r"$T - T_0$ (K)",
          title=r"(b) instantaneous $e^{-\mu x}$ source (1/$\mu$ = 0.5 $\mu$m)")
ax[1].legend(title="lines: exact, circles: solver", fontsize=9)

# (c) Stefan problem: melt front
rho, ks, kl, cs, cl, Lf, Tm, Ti, Tw = 1000.0, 2.0, 1.0, 1000.0, 1500.0, 1e5, 1000.0, 900.0, 1200.0
als, all_ = ks / (rho * cs), kl / (rho * cl)
nu = np.sqrt(all_ / als)
StL, StS = cl * (Tw - Tm) / Lf, cs * (Tm - Ti) / Lf
lam = brentq(lambda l: StL * np.exp(-l**2) / erf(l) - StS / nu * np.exp(-(nu * l)**2) / erfc(nu * l) - l * np.sqrt(np.pi), 1e-6, 5)
m2 = constant_material("stefan", rho, ks, cs, Tm=Tm, L_f=Lf, dTm=0.5, k_liq=kl, cp_liq=cl)
s = ThermalSolver(Mesh.build([Layer(m2, 0.02, 2e-6, 1.02, dx_max=10e-6)], Ti), Ti, front=FrontBC("dirichlet", T_wall=Tw))
fr = []
s.run(1.0, dt0=1e-9, dT_target=2.0, record=lambda sv: fr.append((sv.t, float(np.sum(sv.liquid_fraction() * sv.dx())))))
t, X = np.array(fr).T
ax[2].plot(t[::20], X[::20] * 1e3, "o", ms=3, mfc="none", label="solver (enthalpy)")
ax[2].plot(t, 2 * lam * np.sqrt(all_ * t) * 1e3, "k-", lw=1, label="Neumann solution")
curves.update(c_t_s=t[::20], c_X_mm=X[::20] * 1e3, c_t_exact_s=t, c_X_exact_mm=2 * lam * np.sqrt(all_ * t) * 1e3)
ax[2].set(xlabel="time (s)", ylabel="melt depth (mm)", title="(c) two-phase Stefan problem")
ax[2].legend()

fig.suptitle("Verification: 1D enthalpy solver vs analytical solutions (test materials only)", y=1.02)
fig.tight_layout()
out = ROOT / "runs" / "verification"
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"verification.{ext}", dpi=150, bbox_inches="tight")
np.savez(out / "verification_curves.npz", **curves)
print("saved", out / "verification.png", "and", out / "verification_curves.npz")
