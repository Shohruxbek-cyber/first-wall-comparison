"""Tests for the tungsten-grade registry and the alloy option (alloy study, 2026-10-13).

The registry must load with every property carrying a unit and either a source or an explicit
flag; W_pure must be bit-identical to the baseline tungsten(); an unsourced conductivity must
refuse to run silently; and two physics checks: zero flux -> no change (with the recrystallization
bookkeeping giving zero / the exact constant-temperature value), and the surface-temperature rise
under constant flux scaling as 1/sqrt(k) (dT = 2 q sqrt(t) / sqrt(pi k rho c))."""
import numpy as np
import pytest

from ife_wall import alloys
from ife_wall.diagnostics import RxSpec, ShotRecorder, _time_above, _time_above_vec
from ife_wall.library import tungsten
from ife_wall.materials import constant_material
from ife_wall.scenarios import SCENARIOS, Options, _w_material, w_case
from ife_wall.thermal import FrontBC, Layer, Mesh, ThermalSolver

T_GRID = np.linspace(300.0, 6000.0, 58)


def test_1_registry_loads_every_property_has_unit_and_source_or_flag():
    assert {"W_pure", "W_K", "W_ODS_La2O3", "W_ODS_Y2O3"} <= set(alloys.GRADES)
    assert alloys.RUNNABLE == ["W_pure", "W_K", "W_ODS_La2O3"]
    rows = alloys.property_rows()
    assert len(rows) > 40
    for r in rows:
        assert r["unit"], r
        if r["sourced"]:
            for key in r["source"].split(","):
                assert key.strip() in alloys.REFS, (r["grade"], r["property"], key)
        else:  # flagged: either an explicit None (TODO_SOURCE) or a stated modelling assumption
            assert r["value"] is None or (r["confidence"] == "assumption" and r["note"]), r
    for k, v in alloys.REFS.items():
        assert "doi:" in v or "not independently verified" in v or "repository" in v, k
    flagged = {(r["grade"], r["property"]) for r in alloys.unsourced()}
    assert {("W_ODS_La2O3", "k_factor"), ("W_ODS_Y2O3", "k_factor"), ("W_ODS_Y2O3", "T_rx"),
            ("W_pure", "he_threshold_factor"), ("W_K", "he_threshold_factor"), ("W_K", "Tm"), ("W_ODS_La2O3", "vapor"),
            ("W_pure", "emissivity")} <= flagged
    assert alloys.GRADES["W_pure"].Tm.value == 3695.0 and alloys.GRADES["W_pure"].diffusivity_factor() == 1.0
    assert alloys.GRADES["W_ODS_La2O3"].diffusivity_factor() is None
    for key in alloys.RUNNABLE:
        g = alloys.GRADES[key]
        assert g.T_rx.sourced and g.T_rx.unit == "K" and g.T_rx.value > 1400.0
        assert alloys.recrystallization_thresholds(key) == (g.T_rx.value - 100.0, g.T_rx.value, g.T_rx.value + 100.0)
    assert alloys.Q_RX.sourced and 330e3 < alloys.q_rx_mid() < 340e3


def test_2_W_pure_bit_identical_to_tungsten():
    base = tungsten()
    W = _w_material(Options(w_material="W_pure"))
    assert np.array_equal(W.k(T_GRID), base.k(T_GRID))
    assert np.array_equal(W.rho(T_GRID), base.rho(T_GRID))
    assert np.array_equal(W._T, base._T) and np.array_equal(W._h, base._h)
    assert Options().w_material == "W_pure" and Options().alloy_k_proxy is None
    assert alloys.conductivity_factor(Options()) == 1.0
    assert "SOURCED" in alloys.thermal_status(Options())


def test_3_W_K_thermal_identical_to_pure():
    base = tungsten()
    W = _w_material(Options(w_material="W_K"))
    assert np.array_equal(W.k(T_GRID), base.k(T_GRID))
    assert alloys.GRADES["W_K"].k_factor.sourced and alloys.GRADES["W_K"].k_factor.value == 1.0


def test_4_unsourced_conductivity_refuses_without_proxy():
    with pytest.raises(ValueError):
        alloys.conductivity_factor(Options(w_material="W_ODS_La2O3"))
    with pytest.raises(KeyError):
        alloys.conductivity_factor(Options(w_material="W_unknown"))
    with pytest.raises(ValueError):  # literature-only entries cannot be run, proxy or not
        alloys.conductivity_factor(Options(w_material="W_ODS_Y2O3", alloy_k_proxy=0.9))
    base = tungsten()
    with pytest.warns(UserWarning):
        W = _w_material(Options(w_material="W_ODS_La2O3", alloy_k_proxy=0.8))
    for T in (600.0, 1500.0, 3000.0, 4000.0):  # the proxy multiplies the whole curve (solid and liquid)
        assert abs(W.k(np.array(T)) / base.k(np.array(T)) - 0.8) < 1e-12
    assert "PROXY" in alloys.thermal_status(Options(w_material="W_ODS_La2O3", alloy_k_proxy=0.8))
    with pytest.warns(UserWarning):  # stacks with k_scale and the solid-only damage factor
        W2 = _w_material(Options(w_material="W_ODS_La2O3", alloy_k_proxy=0.8, k_scale=0.5), 0.5)
    assert abs(W2.k(np.array(1500.0)) / base.k(np.array(1500.0)) - 0.2) < 1e-12
    assert abs(W2.k(np.array(4000.0)) / base.k(np.array(4000.0)) - 0.4) < 1e-12


def test_5_time_above_vectorized_matches_scalar():
    rng = np.random.default_rng(3)
    T0, T1 = rng.uniform(1000, 2000, 500), rng.uniform(1000, 2000, 500)
    v = _time_above_vec(2.0, T0, T1, 1500.0)
    ref = np.array([_time_above(0.0, 2.0, a, b, 1500.0) for a, b in zip(T0, T1)])
    assert np.allclose(v, ref, rtol=0, atol=1e-15)


@pytest.mark.parametrize("T0", [600.0, 1600.0])
def test_6_zero_flux_no_change(T0):
    with pytest.warns(UserWarning):
        s = w_case(SCENARIOS["Experimental"], Options(w_material="W_ODS_La2O3", alloy_k_proxy=0.8, T_init=T0))
    s.sources = []
    s._frac_cache = None
    Tstart = s.T.copy()
    rx = RxSpec(thresholds=(1523.15,), Q=(336e3,), T_ref=1523.15)
    rec = ShotRecorder(s, 0.0, None, rx=rx)
    t_end = 20e-6
    s.run(t_end, dt0=1e-12, dT_target=5.0, record=rec.update)
    assert abs(s.t - t_end) < 1e-18
    assert np.max(np.abs(s.T - Tstart)) < 1e-9
    r = rec.results()["recrystallization"]
    expect_above = t_end if T0 >= 1523.15 else 0.0
    assert abs(r["surface"]["time_above_s"]["1523.15"] - expect_above) < 1e-15
    expect_teq = t_end * np.exp(-336e3 / 8.314462618 * (1 / T0 - 1 / 1523.15))
    assert abs(r["surface"]["t_eq_s"]["336000"] / expect_teq - 1) < 1e-9
    rows = rec.rx_profile_rows()
    assert rows.shape[1] == 3 and np.allclose(rows[:, 1], expect_above, atol=1e-15)
    assert np.allclose(rows[:, 2] / expect_teq, 1.0, rtol=1e-9)
    assert "recrystallization" not in ShotRecorder(s, 0.0, None).results()  # default: unchanged output


def test_7_surface_rise_scales_as_inverse_sqrt_conductivity():
    q, t_end, rho, cp = 1e9, 1e-6, 19300.0, 140.0
    rise = {}
    for f in (1.0, 0.8):
        K = 100.0 * f
        alpha = K / (rho * cp)
        L = 20 * np.sqrt(alpha * t_end)
        s = ThermalSolver(Mesh.build([Layer(constant_material("test", rho, K, cp), L, L / 2e4, 1.1)], 300.0), 300.0,
                          front=FrontBC("flux", flux=lambda t: q))
        s.run(t_end, dt0=t_end * 1e-6, dT_target=1.0)
        rise[f] = s.surface_temperature() - 300.0
        assert abs(rise[f] / (2 * q / K * np.sqrt(alpha * t_end / np.pi)) - 1) < 0.01
    assert abs(rise[0.8] / rise[1.0] - 1 / np.sqrt(0.8)) < 0.01
