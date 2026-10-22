"""Tests for the damaged-conductivity options (study of 2026-09-29)."""
import numpy as np

from ife_wall.diagnostics import ShotRecorder
from ife_wall.library import tungsten
from ife_wall.scenarios import SCENARIOS, Options, _w_material, interface_index, w_case


def _layer_mats(s):
    return {j: s.materials[j] for j in range(len(s.materials))}


def test_1_defaults_unchanged():
    base = tungsten()
    W = _w_material(Options())
    T = np.linspace(300.0, 6000.0, 20)
    assert np.allclose(W.k(T), base.k(T), rtol=0, atol=0)


def test_2_solid_only_factor():
    base = tungsten()
    W = _w_material(Options(k_irr=0.5), 0.5)
    for T in (1000.0, 3000.0):
        assert abs(W.k(np.array(T)) / base.k(np.array(T)) - 0.5) < 1e-12
    assert abs(W.k(np.array(4000.0)) / base.k(np.array(4000.0)) - 1.0) < 1e-12  # liquid: undamaged
    # k_scale still multiplies on top
    W2 = _w_material(Options(k_scale=0.85), 0.5)
    assert abs(W2.k(np.array(1500.0)) / base.k(np.array(1500.0)) - 0.425) < 1e-12


def test_3_three_layer_skin():
    opt = Options(skin_um=5.0, skin_factor=0.3, k_irr=0.5)
    s = w_case(SCENARIOS["Demo"], opt)
    assert len(s.materials) == 3 and s.layer.max() == 2
    skin_t = float(s.dx()[s.layer == 0].sum())
    assert abs(skin_t / 5e-6 - 1) < 0.01
    xs, ions = s.sources
    assert list(ions.layer_elements) == ["W", "W", "Fe"]
    base = tungsten()
    T = np.array(1500.0)
    assert abs(s.materials[0].k(T) / base.k(T) - 0.3) < 1e-12
    assert abs(s.materials[1].k(T) / base.k(T) - 0.5) < 1e-12
    i = interface_index(s)
    assert s.layer[i] == 1 and s.layer[i + 1] == 2
    # two-layer case: helper equals the old argmax(layer == 1) - 1
    s2 = w_case(SCENARIOS["Demo"], Options())
    assert interface_index(s2) == int(np.argmax(s2.layer == 1)) - 1
    # mesh continuity at the skin/bulk boundary
    dx = s.dx()
    j = int(np.argmax(s.layer == 1))
    assert 0.5 < dx[j] / dx[j - 1] < 2.0


def test_4_energy_conservation_three_layers():
    s = w_case(SCENARIOS["Demo"], Options(skin_um=5.0, skin_factor=0.25, k_irr=0.5, T_init=921.1))
    s.run(5e-6, dt0=1e-13, dT_target=5.0)
    assert s.ledger.deposited > 5e4
    assert abs(s.ledger.residual()[1]) < 1e-6


def test_5_he_layer_zero_below_1000K():
    s = w_case(SCENARIOS["Experimental"], Options())
    s.sources = [s.sources[0]]  # x-rays only
    s._frac_cache = None
    rec = ShotRecorder(s, 0.0, None)
    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec.update)
    res = rec.results()
    assert max(r["peak_K"] for r in res["He_layer"]) < 1000.0
    assert all(r["time_above_1000K_us"] == 0.0 for r in res["He_layer"])
