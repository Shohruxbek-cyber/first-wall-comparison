"""Cross-check of the compiled model against the independent NumPy reference implementation (model.HeModel).

(a) right-hand side and Jacobian agree to round-off for a random state (Level 1);
(b) in the NumPy model, whose ledger also counts V-I recombination, I annihilation of He-V, bulk sinks and surface
    losses, the He, vacancy and interstitial balances close to < 1e-9 over several pulsed shots
    (small problem: 60 cells, K = 4, constant T = 1100 K, sinks on).
"""
import numpy as np

from _he_common import record
from ife_wall.he import constants as C
from ife_wall.he.fast import FastHe
from ife_wall.he.grid import make_grid
from ife_wall.he.model import HeModel, N_W


def main():
    edges = make_grid()
    p = C.baseline()
    m, f = HeModel(edges, p), FastHe(edges, p)
    Tc = np.full(m.n, 1100.0)
    m.T_fn = lambda t: Tc
    f.set_cycle([0.0, 1e30], np.tile(Tc, (2, 1)))
    rng = np.random.default_rng(3)
    u = rng.random((m.n, m.S)) * 1e-2
    y_np = np.zeros(m.N); y_np[:m.N0] = u.ravel()
    m.g_He = np.zeros(m.n); m.g_V = np.zeros(m.n)
    yf = np.zeros(f.N); yf[6:] = u.ravel()
    r1 = np.abs(f.rhs(0, yf)[6:] - m.rhs(0, y_np)[:m.N0]).max() / np.abs(m.rhs(0, y_np)[:m.N0]).max()
    J_np = m.jac(0, y_np).toarray()[:m.N0, :m.N0]
    J_nb = f.dense_jac(0, yf)[6:, 6:]
    r2 = np.abs(J_nb - J_np).max() / np.abs(J_np).max()
    # (b) NumPy conservation with all channels active
    e2 = make_grid()[:61]
    p2 = C.baseline(K=4, Eb=C.EB_BASE[:4].copy())
    m2 = HeModel(e2, p2)
    Tc2 = np.full(m2.n, 1100.0)
    m2.T_fn = lambda t: Tc2
    x = m2.x * 1e6
    gHe_peak = 4e4 * np.exp(-((x - 0.3) / 0.15) ** 2)      # appm/s
    gV_peak = 2e4 * np.exp(-x / 0.4)
    y = m2.y0()
    for shot in range(4):
        t0 = shot * 1e-3
        m2.g_He, m2.g_V = gHe_peak, gV_peak
        y, _ = m2.integrate(y, t0, t0 + 2e-5, method="Radau", rtol=1e-9, atol=1e-14, atol_led=1e-14)
        m2.g_He, m2.g_V = 0 * gHe_peak, 0 * gV_peak
        y, _ = m2.integrate(y, t0 + 2e-5, t0 + 1e-3, method="Radau", rtol=1e-9, atol=1e-14, atol_led=1e-14)
    cons = m2.conservation(y)
    rel = {k: abs(v[0]) / v[1] for k, v in cons.items()}
    ok = r1 < 1e-12 and r2 < 1e-12 and max(rel.values()) < 1e-9
    record("test3b_numpy_crosscheck", ok, rhs_rel_diff=r1, jac_rel_diff=r2,
           He_balance=rel["He"], V_balance=rel["V"], I_balance=rel["I"], threshold=1e-9)


if __name__ == "__main__":
    main()
