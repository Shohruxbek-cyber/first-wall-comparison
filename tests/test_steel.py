"""Tests for the steel-behind-the-film sources and the steel-conductivity option (revision of 21 Oct 2026)."""
import numpy as np

from ife_wall.he import damage as dm
from ife_wall.he.sources import HE_SPECIES, split_bins
from ife_wall.he.steel import SteelBehindFilm, TargetDisplacementModel
from ife_wall.library import ferritic_steel
from ife_wall.scenarios import SCENARIOS, Options, li_case, w_case


def test_1_target_model_reproduces_tungsten():
    """With the tungsten constants the general displacement model is the production tungsten model."""
    for sp in ("4He", "Deuterons"):
        gen = TargetDisplacementModel(sp, "W", dm.Z2, dm.A2, dm.ED_NRT)
        ref = dm.displacement_model(sp)
        assert np.allclose(gen.sigma_d, ref.sigma_d, rtol=1e-12, atol=0)


def test_2_helium_falls_with_film_and_vanishes_behind_1mm():
    sub = split_bins(SCENARIOS["Demo"], 24)
    he = [SteelBehindFilm("Demo", d * 1e-6, sub_bins=sub, n_cells=10).fluence(HE_SPECIES) for d in (1, 10, 31.6, 100, 1000)]
    assert all(a >= b for a, b in zip(he, he[1:]))
    assert he[-1] == 0.0 and he[0] > 0.0


def test_3_bare_steel_receives_the_incident_ion_energy():
    sub = split_bins(SCENARIOS["Prototype"], 24)
    sb = SteelBehindFilm("Prototype", 1e-9, sub_bins=sub, n_cells=10)
    incident = sum(s.n * s.E for s in sub)
    assert abs(sb.energy_into_steel() / incident - 1.0) < 1e-3


def test_4_steel_conductivity_default_unchanged():
    T = np.linspace(300.0, 1500.0, 7)
    assert np.all(ferritic_steel().k(T) == 33.0)
    assert np.all(ferritic_steel(k=29.0).k(T) == 29.0)
    assert Options().steel_k is None
    for s in (w_case(SCENARIOS["Demo"], Options()), li_case(SCENARIOS["Demo"], 31.6e-6, 600.0, Options())):
        assert float(s.materials[-1].k(np.array(700.0))) == 33.0
    s = w_case(SCENARIOS["Demo"], Options(steel_k=29.0))
    assert float(s.materials[-1].k(np.array(700.0))) == 29.0
