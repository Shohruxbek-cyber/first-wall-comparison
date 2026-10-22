"""Tests for the optional neutron-heating source (ife_wall.neutron) and the grid-growth options
(revision of 20 Oct 2026).  Both are off, or at the production values, by default."""
import numpy as np
import pytest

from ife_wall.neutron import E_W_REF, NeutronHeatingSource, neutron_arrival
from ife_wall.scenarios import SCENARIOS, Options, li_case, w_case

DEMO, PROTO = SCENARIOS["Demo"], SCENARIOS["Prototype"]


def test_1_defaults_add_no_source_and_keep_the_production_grid():
    s = w_case(DEMO, Options())
    assert len(s.sources) == 2 and not any(isinstance(x, NeutronHeatingSource) for x in s.sources)
    assert len(s.m) == 308 and s.m[0] == pytest.approx(1e-9 * float(s.materials[0].rho(np.array(600.0))), rel=1e-12)
    t = li_case(DEMO, 31.6e-6, 600.0, Options())
    assert len(t.sources) == 2
    for opt in (Options(), Options(grid_growth=1.05, grid_growth_steel=1.15)):
        assert np.array_equal(w_case(DEMO, opt).m, s.m)


def test_2_finer_growth_gives_more_cells():
    assert len(w_case(DEMO, Options(grid_growth=1.025)).m) > len(w_case(DEMO, Options()).m)


def test_3_neutron_energy_per_shot_matches_the_layer_thicknesses():
    assert E_W_REF == pytest.approx(4e6)  # 800 MW/cm^3 x 5 ns = 4 J/cm^3
    s = w_case(DEMO, Options(neutron_heating=True, neutron_steel_factor=0.5))
    src = s.sources[2]
    src.fractions(s)
    # 1 mm of W at 4e6 J/m^3 and 1 cm of steel at half of it, both at the initial density
    assert src.fluence == pytest.approx(4e6 * 1e-3 + 0.5 * 4e6 * 1e-2, rel=1e-9)
    p = w_case(PROTO, Options(neutron_heating=True, neutron_steel_factor=0.0)).sources[2]
    p.fractions(w_case(PROTO, Options()))
    assert p.fluence == pytest.approx(4e6 * 50 / 154 * 1e-3, rel=1e-9)
    li = li_case(DEMO, 31.6e-6, 600.0, Options(neutron_heating=True, neutron_steel_factor=1.0))
    n = li.sources[2]
    fr = n.fractions(li)
    assert fr[li.layer == 0].sum() == 0.0  # the lithium film is not heated
    assert n.fluence == pytest.approx(4e6 * 1e-2, rel=1e-9)
    assert 1.2e-7 < neutron_arrival(6.5) < 1.3e-7


def test_4_short_run_with_neutron_heating_conserves_energy():
    s = w_case(DEMO, Options(neutron_heating=True, neutron_steel_factor=1.0))
    s.run(1e-6, dt0=1e-13, dT_target=5.0)
    s.ledger.stored_final = s.stored_energy()
    assert abs(s.ledger.residual()[1]) < 1e-10
    n = s.sources[2]
    assert n.energy(0.0, 1.0) == pytest.approx(n.fluence, rel=1e-12)
