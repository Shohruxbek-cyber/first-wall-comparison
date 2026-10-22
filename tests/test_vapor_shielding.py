"""Tests for the reduced vapor-shielding option (ife_wall.vapor; revision of 20 Oct 2026).

Model: Section 2.5 of the paper.  The option is off by default, so the existing
results are reproduced by construction (test 1); the full-length regression runs (W Demo and
Li Demo at 31.6 um against the stored production values) are not part of the test suite.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ife_wall.ions import IonSpectrumSource
from ife_wall.li_multishot import li_multishot_case
from ife_wall.scenarios import SCENARIOS, Options, li_case, w_case
from ife_wall.vapor import VaporShieldedSource, _ExtendedMesh
from ife_wall.xray import XraySource

ROOT = Path(__file__).resolve().parents[1]
DEMO = SCENARIOS["Demo"]
DELTA = float(np.geomspace(1e-6, 1e-3, 13)[6])  # 31.6 um, a production thickness


def _li(**kw):
    return li_case(DEMO, DELTA, 600.0, Options(**kw))


def test_1_defaults_build_the_unshielded_sources():
    assert Options().vapor_shielding is False
    for s in (_li(), w_case(DEMO, Options())):
        xs, ions = s.sources
        assert type(xs) is XraySource and type(ions) is IonSpectrumSource
    s = _li(vapor_shielding=True, vapor_f_rr=0.5, vapor_stopping_factor=2.0, vapor_tau_rr=1e-4)
    w = s.sources[1]
    assert isinstance(w, VaporShieldedSource)
    assert (w.f_rr, w.stopping_factor, w.tau_rr) == (0.5, 2.0, 1e-4)


def test_2_zero_column_returns_the_base_fractions_exactly():
    s = _li()
    base = s.sources[1]
    f0 = np.asarray(base.fractions(s)).copy()
    w = VaporShieldedSource(base, f_rr=1.0)
    f1 = w.fractions(s)
    assert np.array_equal(f1[:-1], f0)
    assert f1[-1, 0] == 1.0 and not f1[-1, 1:].any()
    ev = w.energy_components(0.0, 1e-5)
    assert np.array_equal(ev[:-1], base.energy_components(0.0, 1e-5)) and ev[-1] == 0.0


def test_3_vapor_and_wall_shares_partition_the_extended_mesh():
    s = _li()
    base = s.sources[1]
    M = 3e-4
    w = VaporShieldedSource(base, fixed_column=M)
    wall = w.fractions(s)[:-1]
    ext = np.asarray(base.fractions(_ExtendedMesh(M, s.m, s.layer)))
    assert np.array_equal(w.vapor_share, ext[:, 0])
    assert np.array_equal(wall, ext[:, 1:])
    # the stopping factor is an effective column: s = 2 equals twice the column
    w2 = VaporShieldedSource(base, stopping_factor=2.0, fixed_column=M)
    w2.fractions(s)
    w3 = VaporShieldedSource(base, fixed_column=2 * M)
    w3.fractions(s)
    assert np.array_equal(w2.vapor_share, w3.vapor_share)


def test_4_full_per_shot_column_reproduces_the_static_upper_bound():
    p = ROOT / "runs/li_scan/vapour_shielding_bound_Demo.json"
    if not p.exists():  # a fresh copy of the release: the stored output of the vapor_shielding stage
        p = ROOT / "results/vapor_shielding/li_scan/vapour_shielding_bound_Demo.json"
    bound = json.loads(p.read_text(encoding="utf-8"))
    up = bound["upper_full_per_shot_vapour"]
    s = li_case(DEMO, 1e-3, 600.0, Options())  # the bound script's film
    base = s.sources[1]
    w = VaporShieldedSource(base, fixed_column=up["column_kg_m2"])
    w.fractions(s)
    E = base.fluence
    assert float(E @ w.vapor_share / E.sum()) == pytest.approx(up["ion_energy_fraction_stopped"], rel=1e-12)


def test_5_thick_column_stops_everything_and_returns_it_when_f_rr_is_one():
    s = _li()
    base = s.sources[1]
    for f_rr in (0.0, 1.0):
        w = VaporShieldedSource(base, f_rr=f_rr, fixed_column=10.0)  # 10 kg/m^2 of Li, ~2 cm
        fr = w.fractions(s)
        assert w.vapor_share.min() > 1 - 1e-12
        assert fr[:-1].sum() < 1e-12
        ev = w.energy_components(0.0, 1e-3)
        assert ev[-1] == pytest.approx(f_rr * ev[:-1].sum(), rel=1e-12, abs=0.0)


def test_6_short_coupled_run_conserves_energy():
    s = _li(vapor_shielding=True, vapor_f_rr=0.5, vapor_tau_rr=2e-6)
    s.run(5e-6, dt0=1e-13, dT_target=5.0)
    s.ledger.stored_final = s.stored_energy()
    assert abs(s.ledger.residual()[1]) < 1e-10
    w = s.sources[1]
    rec = w.finalize(s.t)
    assert rec["ion_energy_in_vapor_J_m2"] > 0.0 and rec["max_column_kg_m2"] > 0.0
    # stopped = returned + stored + lost, with lost = (1 - f_rr) x stopped
    assert rec["reradiated_to_wall_J_m2"] + rec["stored_in_vapor_J_m2"] == pytest.approx(
        0.5 * rec["ion_energy_in_vapor_J_m2"], rel=1e-9)
    # the solver books the returned energy as incident and deposited: its incident energy
    # exceeds the true one (x-rays plus ions) by exactly the returned energy
    E_true = s.sources[0].fluence + rec["ion_energy_incident_J_m2"]
    assert s.ledger.incident - rec["reradiated_to_wall_J_m2"] == pytest.approx(E_true, rel=1e-12)


def test_7_retried_steps_are_not_counted():
    s = _li()
    w = VaporShieldedSource(s.sources[1], f_rr=1.0, fixed_column=2e-4)
    w.fractions(s)
    t0, t1, t1b, t2 = 1e-6, 1.2e-6, 1.05e-6, 1.3e-6
    w.energy_components(t0, t1)   # attempt, rejected
    e_ok = w.energy_components(t0, t1b)  # retry, accepted
    w.energy_components(t1b, t2)  # next step: commits the retry only
    assert w.E_incident == pytest.approx(float(e_ok[:-1].sum()), rel=1e-15)
    rec = w.finalize(t2)
    assert rec["ion_energy_incident_J_m2"] == pytest.approx(float(w.base.energy(t0, t2)), rel=1e-12)


def test_8_periodic_tungsten_runs_refuse_the_option_and_lithium_shots_restart_the_column():
    with pytest.raises(ValueError):
        w_case(DEMO, Options(vapor_shielding=True), n_shots=2)
    s = li_multishot_case(DEMO, DELTA, 600.0, Options(vapor_shielding=True))
    w = s.sources[1]
    s.mass_evaporated = 4e-4  # as after a shot
    w.start_shot(s)
    fr = w.fractions(s)
    assert w.column == 0.0 and not w.vapor_share.any()
    s.mass_evaporated = 4e-4 + 1e-4
    w.fractions(s)
    assert w.column == pytest.approx(1e-4, rel=1e-12)


def test_10_coupled_column_follows_the_evaporated_mass():
    """In a coupled Li Demo run the column used for the deposition in each accepted step differs
    from the mass evaporated before that step by at most the solver's refresh threshold (1 % of
    the surface-cell mass).  (The mass evaporated within a step enters from the next step on.)"""
    s = _li(vapor_shielding=True)
    w = s.sources[1]
    prev, lag = [0.0], []

    def rec(sv):
        lag.append(abs(w.column - prev[0]))  # column used in this step vs the mass at its start
        prev[0] = sv.mass_evaporated

    s.run(4e-6, dt0=1e-13, dT_target=5.0, record=rec)
    assert s.mass_evaporated > 1e-5
    assert max(lag) <= 0.01 * s.m_target0[0] * (1 + 1e-9)


def test_11_delayed_return_follows_the_exponential_store():
    """Constant stopped power P for a time t0, then none: the store and the returned energy
    follow E_st = f P tau (1 - exp(-t/tau)) and its free decay (Eq. 3), step by step."""
    s = _li()
    tau, f, P, t_on = 2e-6, 0.7, 1.0e9, 5e-6

    class Const:
        bins = [None]
        fluence = np.array([1.0])
        t_first, t_last = 0.0, 1.0

        def energy_components(self, a, b):
            return np.array([P * max(min(b, t_on) - a, 0.0)])

        def fractions(self, mesh):
            fr = np.zeros((1, len(mesh.m)))
            fr[0, 0] = 1.0  # everything stopped in the first (vapor) cell
            return fr

        def total_fluence(self):
            return 1.0

    w = VaporShieldedSource(Const(), f_rr=f, tau_rr=tau, fixed_column=1e-3)
    w.fractions(s)
    ts = np.linspace(0.0, 2 * t_on, 401)
    for a, b in zip(ts[:-1], ts[1:]):
        w.energy_components(a, b)
    w.finalize(ts[-1])
    E_on = f * P * tau * (1 - np.exp(-t_on / tau))  # store at the end of the source
    store_end = E_on * np.exp(-(ts[-1] - t_on) / tau)
    assert w.E_store == pytest.approx(store_end, rel=1e-9)
    assert w.E_reradiated + w.E_store == pytest.approx(f * P * t_on, rel=1e-9)


def test_12_small_columns_stop_a_small_positive_share_that_grows_with_the_column():
    s = _li()
    shares = []
    for M in (1e-9, 1e-8, 1e-7, 1e-6):
        w = VaporShieldedSource(s.sources[1], fixed_column=M)
        w.fractions(s)
        shares.append(float(s.sources[1].fluence @ w.vapor_share))
    assert shares[0] > 0 and all(b > a for a, b in zip(shares, shares[1:]))


def test_13_tungsten_column_is_negligible_even_if_present_from_the_start():
    """The Demo W shot evaporates about 2e-11 kg/m^2; that column stops < 1e-8 of the ion energy."""
    s = w_case(DEMO, Options())
    w = VaporShieldedSource(s.sources[1], fixed_column=2.2e-11)
    w.fractions(s)
    E = s.sources[1].fluence
    assert 0 < float(E @ w.vapor_share / E.sum()) < 1e-8


class _MonoIons:
    """One monoenergetic ion component with constant power over [t1, t2] and a constant
    stopping per unit areal mass S (range E0/S), for the analytic test below."""

    def __init__(self, E0, S, F, t1, t2):
        self.E0, self.S, self.F, self.t1, self.t2 = E0, S, F, t1, t2
        self.bins = [None]
        self.fluence = np.array([F])
        self.t_first, self.t_last = t1, t2

    def energy_components(self, t0, t1):
        a, b = max(t0, self.t1), min(t1, self.t2)
        return np.array([self.F * max(b - a, 0.0) / (self.t2 - self.t1)])

    def fractions(self, mesh):
        b = np.r_[0.0, np.cumsum(mesh.m)]
        lost = np.minimum(self.S * b, self.E0)  # energy lost after crossing areal mass b
        return (np.diff(lost) / self.E0)[None, :]

    def total_fluence(self):
        return self.F

    def dt_limit(self, t):
        return np.inf

    def next_event(self, t):
        return np.inf


def test_9_linear_column_growth_matches_the_analytic_integral():
    """Column M(t) = a t ahead of monoenergetic ions with constant stopping S: the stopped energy
    is F/(t2 - t1) * integral of S a t / E0 dt = F S a (t2^2 - t1^2) / (2 E0 (t2 - t1)), exactly
    integrable; the wrapper samples the column at the start of each step (left Riemann sum)."""
    E0, S, F, t1, t2, a = 1.0e-13, 1.0e-10, 5.0e4, 1e-6, 3e-6, 200.0  # J; J m^2/kg; J/m^2; s; s; kg m^-2 s^-1
    base = _MonoIons(E0, S, F, t1, t2)
    w = VaporShieldedSource(base, f_rr=0.5)

    class FakeSolver:
        m = np.full(50, 1e-5)
        layer = np.zeros(50, int)
        mass_evaporated = 0.0

    sv = FakeSolver()
    n = 4000
    ts = np.linspace(t1, t2, n + 1)
    for t0, t_1 in zip(ts[:-1], ts[1:]):
        sv.mass_evaporated = a * t0
        w.fractions(sv)
        w.energy_components(t0, t_1)
    rec = w.finalize(t2)
    exact = F * S * a * (t2 ** 2 - t1 ** 2) / (2 * E0 * (t2 - t1))
    assert rec["ion_energy_in_vapor_J_m2"] == pytest.approx(exact, rel=2e-3)  # left-sum error ~ 1/n
    assert rec["reradiated_to_wall_J_m2"] == pytest.approx(0.5 * rec["ion_energy_in_vapor_J_m2"], rel=1e-12)
