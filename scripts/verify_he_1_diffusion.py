"""Verification test 1: pure He diffusion vs the exact solution, constant T.  Pass: max error < 1 %.

Initial condition: Gaussian pulse (centre x0, width s0) of He, absorbing free surface at x = 0, no flux at the
back, no vacancies (so no reactions).  Exact solution (method of images, valid while the pulse is far from the
back face):  C = M/sqrt(2 pi s^2) [exp(-(x-x0)^2/2s^2) - exp(-(x+x0)^2/2s^2)],  s^2 = s0^2 + 2 D t.
Run on (a) a uniform 25 nm grid and (b) the production grid with the pulse placed in its fine region.
"""
import numpy as np

from _he_common import record, style
from ife_wall.he import constants as C
from ife_wall.he.fast import FastHe
from ife_wall.he.model import NC


def run(edges, x0, s0, times, T=1000.0):
    p = C.baseline(K=1, Eb=np.array([4.05]), sink_V=0.0, sink_I=0.0)
    fm = FastHe(edges, p)
    fm.set_cycle([0.0, 1e30], np.full((2, fm.n), T))
    D = p.D0_He * np.exp(-p.Em_He / (8.617333262e-5 * T)) * 1e12      # um^2/s
    x = fm.x * 1e6
    c0 = np.exp(-(x - x0) ** 2 / (2 * s0 ** 2)) / np.sqrt(2 * np.pi * s0 ** 2)      # unit mass (appm um)
    # cell-averaged initial condition (exact integral of the Gaussian over each cell)
    from scipy.special import erf
    e = fm.edges * 1e6
    cell = 0.5 * (erf((e[1:] - x0) / (np.sqrt(2) * s0)) - erf((e[:-1] - x0) / (np.sqrt(2) * s0))) / (e[1:] - e[:-1])
    M = 0.05  # appm um
    y = np.zeros(fm.N)
    y[6:].reshape(fm.n, fm.S)[:, 0] = M * cell
    mass0 = (M * cell * (e[1:] - e[:-1])).sum()     # discrete initial mass (the Gaussian tail below x = 0 is cut off)
    out = []
    t_prev = 0.0
    for t in times:
        y = fm.integrate(y, t_prev, t, rtol=1e-8, atol=1e-14)
        t_prev = t
        s2 = s0 ** 2 + 2 * D * t
        ex = M / np.sqrt(2 * np.pi * s2) * (np.exp(-(x - x0) ** 2 / (2 * s2)) - np.exp(-(x + x0) ** 2 / (2 * s2)))
        num = y[6:].reshape(fm.n, fm.S)[:, 0]
        # compare cell averages of the exact solution (fine quadrature) with the numerical cell values
        xs = np.linspace(0, 1, 9)[None, :] * (e[1:] - e[:-1])[:, None] + e[:-1][:, None]
        exq = (M / np.sqrt(2 * np.pi * s2) * (np.exp(-(xs - x0) ** 2 / (2 * s2)) - np.exp(-(xs + x0) ** 2 / (2 * s2)))).mean(1)
        err = np.abs(num - exq).max() / exq.max()
        mass_err = abs((num * (e[1:] - e[:-1])).sum() + y[0] - mass0) / mass0
        out.append((t, err, mass_err, x, num, exq))
    return D, out


def main():
    plt = style()
    results = {}
    D, out_a = run(np.linspace(0, 40e-6, 1601), x0=1.5, s0=0.15, times=[2e-6, 5e-6, 1e-5, 2e-5])
    errs_a = [o[1] for o in out_a]
    from ife_wall.he.grid import make_grid
    D, out_b = run(make_grid(), x0=0.12, s0=0.04, times=[2e-8, 5e-8, 1e-7, 2e-7])
    errs_b = [o[1] for o in out_b]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a_, out, ttl in ((ax[0], out_a, "uniform 25 nm grid"), (ax[1], out_b, "production grid")):
        for (t, err, me, x, num, ex), col in zip(out, ["C0", "C1", "C2", "C3"]):
            a_.plot(x, ex, "-", color=col, lw=2, label=f"exact, t = {t * 1e6:g} $\\mu$s")
            a_.plot(x, num, "o", color=col, ms=3)
        a_.set(xlabel="depth ($\\mu$m)", ylabel="He (appm)", title=ttl, xlim=(0, 3.2 if a_ is ax[0] else 0.6))
        a_.legend(fontsize=9)
    fig.suptitle("Test 1: He diffusion, numerical (dots) vs exact (lines)")
    fig.savefig(str(__import__("_he_common").VER / "test1_diffusion.pdf"))
    worst = max(errs_a + errs_b)
    record("test1_diffusion", worst < 0.01, max_error_uniform=max(errs_a), max_error_production_grid=max(errs_b),
           max_mass_error=max(o[2] for o in out_a + out_b), threshold=0.01,
           note="error relative to the peak of the exact solution (cell averages); T = 1000 K, D = %.3e um^2/s" % D)


if __name__ == "__main__":
    main()
