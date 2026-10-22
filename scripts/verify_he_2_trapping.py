"""Verification test 2: trapping limit.  A He pulse diffusing through a uniform density Nt of immobile traps
(vacancies, one He per trap, K = 1) must spread with the Oriani effective diffusivity
    D_eff = D / (1 + (N_T/N_L) exp(Eb/kT)),     N_L = W atom density,
valid when trapping/detrapping is fast compared with diffusion and the traps are far from saturation.
With nu = 4 pi r_c D0 N_L the detrapping prefactor of the model equals the one implied by the detailed balance of the
diffusion-limited capture rate, so the model must reproduce the textbook formula with no fitted parameter.
Pass: error < 2 % (profile of total He, relative to its peak), at two temperatures.
"""
import numpy as np

from _he_common import record, style, VER
from ife_wall.he import constants as C
from ife_wall.he.fast import FastHe
from ife_wall.he.model import NC, Params, N_W
from scipy.special import erf

KB = 8.617333262e-5


def run(T, Eb, Nt_appm, times, x0=1.5, s0=0.2):
    nu = 4 * np.pi * C.R_C * C.D0_HE * N_W            # = rate prefactor consistent with the capture rate (detailed balance)
    p = Params(K=1, D0_He=C.D0_HE, Em_He=C.EM_HE, D0_V=1e-30, Em_V=2.0, D0_I=1e-30, Em_I=1.0, r_c=C.R_C, r_iv=C.R_IV,
               nu=nu, Eb=np.array([Eb]), ann_hev=False)
    edges = np.linspace(0, 40e-6, 1601)
    fm = FastHe(edges, p)
    fm.set_cycle([0.0, 1e30], np.full((2, fm.n), T))
    kT = KB * T
    D = C.D0_HE * np.exp(-C.EM_HE / kT) * 1e12
    theta = Nt_appm * 1e-6 * np.exp(Eb / kT)          # Oriani: (N_T/N_L) exp(Eb/kT)
    Deff = D / (1 + theta)
    e = fm.edges * 1e6
    cell = 0.5 * (erf((e[1:] - x0) / (np.sqrt(2) * s0)) - erf((e[:-1] - x0) / (np.sqrt(2) * s0))) / (e[1:] - e[:-1])
    M = 0.02                                          # total He (appm um) << trap capacity
    tot = M * cell
    He = tot / (1 + theta)
    HV1 = theta * He
    u0 = np.zeros((fm.n, fm.S))
    u0[:, 0] = He; u0[:, 3] = HV1; u0[:, 1] = Nt_appm - HV1
    y = np.zeros(fm.N); y[6:] = u0.ravel()
    res = []
    tp = 0.0
    for t in times:
        y = fm.integrate(y, tp, t, rtol=1e-8, atol=1e-14); tp = t
        u = y[6:].reshape(fm.n, fm.S)
        total = u[:, 0] + u[:, 3]
        x = fm.x * 1e6
        s2 = s0 ** 2 + 2 * Deff * t
        xs = np.linspace(0, 1, 9)[None, :] * (e[1:] - e[:-1])[:, None] + e[:-1][:, None]
        ex = (M / np.sqrt(2 * np.pi * s2) * (np.exp(-(xs - x0) ** 2 / (2 * s2)) - np.exp(-(xs + x0) ** 2 / (2 * s2)))).mean(1)
        res.append((t, np.abs(total - ex).max() / ex.max(), x, total, ex))
    return D, Deff, theta, res


def main():
    plt = style()
    cases = [(1000.0, 0.90, 50.0, [2e-6, 5e-6, 1e-5]), (1400.0, 1.20, 200.0, [5e-7, 1e-6, 2e-6])]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    worst, info = 0.0, []
    for a_, (T, Eb, Nt, times) in zip(ax, cases):
        D, Deff, theta, res = run(T, Eb, Nt, times)
        for (t, err, x, tot, ex), col in zip(res, ["C0", "C1", "C2"]):
            a_.plot(x, ex, "-", color=col, lw=2, label=f"Oriani, t = {t * 1e6:g} $\\mu$s")
            a_.plot(x, tot, "o", color=col, ms=3)
            worst = max(worst, err)
        info.append(dict(T=T, Eb=Eb, Nt_appm=Nt, theta=theta, D_um2_s=D, Deff_um2_s=Deff, errors=[r[1] for r in res]))
        a_.set(xlabel="depth ($\\mu$m)", ylabel="mobile + trapped He (appm)", title=f"T = {T:.0f} K, Eb = {Eb} eV, $D_\\mathrm{{eff}}/D$ = {Deff / D:.3f}", xlim=(0, 3.5))
        a_.legend(fontsize=9)
    fig.suptitle("Test 2: diffusion with trapping, numerical (dots) vs Oriani effective diffusivity (lines)")
    fig.savefig(str(VER / "test2_trapping.pdf"))
    record("test2_trapping_oriani", worst < 0.02, max_error=worst, threshold=0.02, cases=info,
           note="K = 1, immobile traps, nu = 4 pi r_c D0 N_W (detailed balance), He << trap capacity")


if __name__ == "__main__":
    main()
