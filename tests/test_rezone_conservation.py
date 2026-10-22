"""Direct test of the front-layer remap ("rezoning") claimed in Section 3.1 of the manuscript to
preserve mass and enthalpy exactly (revision of 20 Oct 2026).

ThermalSolver._rezone_front_layer remaps the front layer, after the surface cell has lost mass
by evaporation, onto the original mass grid measured from the new surface, using a
piecewise-constant remap of the cumulative enthalpy.  "Exactly" can only mean to floating-point
rounding: the test asserts the total mass and the total enthalpy of the layer, and of the whole
domain, before and after the remap agree to a few units in the last place, and checks the new
grid against the original one."""
import numpy as np

from ife_wall.materials import Vapor, constant_material
from ife_wall.thermal import Layer, Mesh, ThermalSolver


def _solver(seed=0):
    vap = Vapor(p_sat=lambda T: np.zeros_like(np.asarray(T, float)), L_v=lambda T: np.full_like(np.asarray(T, float), 1e6),
                molar_mass=0.1)
    front = constant_material("front", 1000.0, 50.0, 500.0, vapor=vap)
    back = constant_material("back", 8000.0, 30.0, 500.0)
    mesh = Mesh.build([Layer(front, 50e-6, 10e-9, 1.05, dx_first_back=1e-6), Layer(back, 1e-3, 1e-6, 1.15)], 600.0)
    s = ThermalSolver(mesh, 600.0)
    rng = np.random.default_rng(seed)
    s.H = s.H + rng.uniform(0.0, 4e5, s.H.shape)  # non-uniform enthalpy so the remap is non-trivial
    return s


def _totals(s):
    f = s.layer == 0
    return s.m[f].sum(), (s.m[f] * s.H[f]).sum(), s.m.sum(), (s.m * s.H).sum()


def test_rezone_preserves_mass_and_enthalpy():
    for seed in range(5):
        s = _solver(seed)
        # remove 60 % of the surface cell (below the 0.5 threshold) as evaporation would
        s.m[0] *= 0.4
        before = _totals(s)
        n_before = int((s.layer == 0).sum())
        s._rezone_front_layer()
        after = _totals(s)
        for b, a in zip(before, after):
            assert abs(a - b) <= 8 * np.finfo(float).eps * abs(b), (seed, b, a, (a - b) / b)
        assert int((s.layer == 0).sum()) <= n_before
        assert s.n_rezones == 1


def test_rezone_grid_is_original_grid_from_new_surface():
    s = _solver(1)
    s.m[0] *= 0.3
    M = s.m[s.layer == 0].sum()
    s._rezone_front_layer()
    m_new = s.m[s.layer == 0]
    n = len(m_new)
    # every cell but the last is a cell of the original front-layer grid, in order from the new surface;
    # the new masses are differences of cumulative sums, so they match to the rounding of those sums
    # (a few eps times the layer mass), not bitwise
    assert np.allclose(m_new[:-1], s.m_target0[: n - 1], rtol=0.0, atol=16 * np.finfo(float).eps * M)
    # the last cell takes the remainder of the layer's mass
    assert abs(m_new.sum() - M) <= 8 * np.finfo(float).eps * M
