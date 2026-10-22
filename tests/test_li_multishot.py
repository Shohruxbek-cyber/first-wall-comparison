"""Multi-shot Li film (ife_wall.li_multishot, added 2026-10-06): the new path leaves the
single-shot state untouched, and the film reset restores the grid and books the energy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ife_wall.li_multishot import li_multishot_case, reset_film, steel_index  # noqa: E402
from ife_wall.scenarios import SCENARIOS, li_case  # noqa: E402

DELTA, T_LI = 31.6227766e-6, 600.0


def test_1_baseline_state_identical_to_li_case():
    a = li_case(SCENARIOS["Demo"], DELTA, T_LI)
    b = li_multishot_case(SCENARIOS["Demo"], DELTA, T_LI, T_coolant=T_LI)
    assert np.array_equal(a.m, b.m) and np.array_equal(a.H, b.H) and np.array_equal(a.layer, b.layer)
    assert (a.back.kind, a.back.h, a.back.T_coolant) == (b.back.kind, b.back.h, b.back.T_coolant)
    assert a.ledger.stored_initial == b.ledger.stored_initial
    assert b.ledger.removed_with_film == 0.0


def test_2_steel_starts_at_coolant_temperature():
    b = li_multishot_case(SCENARIOS["Demo"], DELTA, 800.0, T_coolant=600.0)
    T = b.T
    assert np.allclose(T[b.layer == 0], 800.0, atol=0.01) and np.allclose(T[b.layer == 1], 600.0, atol=0.01)
    assert b.back.T_coolant == 600.0


def test_3_reset_film_restores_grid_and_books_energy():
    s = li_multishot_case(SCENARIOS["Demo"], DELTA, T_LI)
    m0, H0, layer0 = s.m.copy(), s.H.copy(), s.layer.copy()
    li = s.layer == 0
    # perturb the film (hot, half the cells evaporated) and the steel front
    s.H[li] = s.materials[0].h_of_T(np.full(li.sum(), 1500.0))
    keep = np.r_[np.ones(li.sum() // 2, bool), np.zeros(li.sum() - li.sum() // 2, bool), np.ones((~li).sum(), bool)]
    s.m, s.H, s.layer = s.m[keep], s.H[keep], s.layer[keep]
    s.H[int(np.argmax(s.layer == 1))] = float(s.materials[1].h_of_T(np.array([900.0]))[0])
    E_before = s.stored_energy()
    removed = reset_film(s, T_LI)
    assert np.array_equal(s.m, m0) and np.array_equal(s.layer, layer0)
    assert np.array_equal(s.H[s.layer == 0], H0[layer0 == 0])
    assert np.allclose(s.T[s.layer == 0], T_LI, atol=0.01)
    assert abs(E_before - removed - s.stored_energy()) < 1e-9 * abs(E_before)
    assert s.ledger.removed_with_film == removed and s._frac_cache is None
    assert steel_index(s) == int(np.argmax(layer0 == 1))


def _perturbed(s):
    li = s.layer == 0
    s.H[li] = s.materials[0].h_of_T(np.full(li.sum(), 900.0))
    keep = np.r_[np.ones(li.sum() - 3, bool), np.zeros(3, bool), np.ones((~li).sum(), bool)]  # 3 cells evaporated
    s.m, s.H, s.layer = s.m[keep], s.H[keep], s.layer[keep]
    return s


def test_4_partial_renewal():
    """renew_film (22 Oct 2026, M2): fraction 1 is reset_film; a partial renewal mixes the retained film's enthalpy
    with fresh lithium on the original grid and books the exchanged enthalpy."""
    from ife_wall.li_multishot import renew_film
    a = _perturbed(li_multishot_case(SCENARIOS["Demo"], DELTA, T_LI))
    b = _perturbed(li_multishot_case(SCENARIOS["Demo"], DELTA, T_LI))
    ra = reset_film(a, T_LI)
    rb, Tb = renew_film(b, T_LI, 1.0)
    assert np.array_equal(a.m, b.m) and np.array_equal(a.layer, b.layer)
    assert np.allclose(a.H, b.H, rtol=1e-14, atol=0.0) and abs(ra - rb) <= 1e-9 * abs(ra) and abs(Tb - T_LI) < 1e-6
    c = _perturbed(li_multishot_case(SCENARIOS["Demo"], DELTA, T_LI))
    li = c.layer == 0
    M_old, E_old, M0 = c.m[li].sum(), (c.m[li] * c.H[li]).sum(), c.m_target0.sum()
    E_before = c.stored_energy()
    r, Tf = renew_film(c, T_LI, 0.1)
    h600 = float(c.materials[0].h_of_T(np.array([T_LI]))[0])
    h_mix = (0.9 * E_old + (M0 - 0.9 * M_old) * h600) / M0
    assert np.allclose(c.H[c.layer == 0], h_mix, rtol=1e-14) and np.array_equal(c.m[c.layer == 0], c.m_target0)
    assert 600.0 < Tf < 900.0 and abs(E_before - r - c.stored_energy()) < 1e-9 * abs(E_before)
