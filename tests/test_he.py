"""Fast regression tests of the He-retention module (src/ife_wall/he).  The full verification is in scripts/verify_he_*.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ife_wall.he import constants as C  # noqa: E402
from ife_wall.he import damage as dm  # noqa: E402
from ife_wall.he.fast import FastHe  # noqa: E402
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.model import HeModel  # noqa: E402
from ife_wall.he.sources import ShotSources  # noqa: E402


def test_grid():
    e = make_grid()
    assert abs(e[-1] - 1e-3) < 1e-12
    assert np.diff(e)[0] == 1e-9
    assert (e <= 10e-6).sum() - 1 >= 100          # at least 100 cells in the first 10 um
    assert (e <= 50e-6).sum() - 1 > 140


def test_nrt_formula():
    assert dm.nu_nrt(50.0, 90.0) == 0.0
    assert dm.nu_nrt(100.0, 90.0) == 1.0
    assert abs(float(dm.nu_nrt(1000.0, 90.0)) - 0.8 * 1000.0 / 180.0) < 1e-12
    assert 0.2 < float(dm.nu_arc(1e4, 70.0) / dm.nu_nrt(1e4, 70.0)) < 0.3   # xi(10 keV) = 0.21


def test_compiled_matches_numpy_reference():
    e = make_grid()
    p = C.baseline()
    m, f = HeModel(e, p), FastHe(e, p)
    Tc = np.full(m.n, 1100.0)
    m.T_fn = lambda t: Tc
    f.set_cycle([0.0, 1e30], np.tile(Tc, (2, 1)))
    rng = np.random.default_rng(0)
    u = rng.random((m.n, m.S)) * 1e-2
    yn = np.zeros(m.N); yn[:m.N0] = u.ravel()
    m.g_He = np.zeros(m.n); m.g_V = np.zeros(m.n)
    yf = np.zeros(f.N); yf[6:] = u.ravel()
    a, b = f.rhs(0, yf)[6:], m.rhs(0, yn)[:m.N0]
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-12


def test_he_source_matches_report_values():
    from ife_wall.he.sources import split_bins
    from ife_wall.ions import N_FUSIONS_REF, ion_spectra_tables
    from ife_wall.scenarios import SCENARIOS
    grid = make_grid()
    # the per-species normalized spectra used before 22 Oct 2026 (reports: 5.58e16 He/m^2 burn, 1.23e17 in all)
    S = ShotSources("Demo", grid, sub_bins=split_bins(SCENARIOS["Demo"], 6, spectra="normalized"))
    assert abs(S.he_per_shot_incident("burn") / 5.58e16 - 1) < 0.005
    assert abs(S.he_per_shot_incident() / 1.23e17 - 1) < 0.01
    # the conserving spectra (production since 22 Oct 2026): 4He = one alpha per fusion, plus the table's 3He
    S = ShotSources("Demo", grid)
    t = ion_spectra_tables("conserving")
    area = 4 * np.pi * SCENARIOS["Demo"].R ** 2
    he3 = sum(t[(g, "3He")]["number"].sum() for g in ("burn", "debris"))
    assert abs(S.he_per_shot_incident() / ((N_FUSIONS_REF + he3) / area) - 1) < 1e-12
    burn = t[("burn", "4He")]["number"].sum() + t[("burn", "3He")]["number"].sum()
    assert abs(S.he_per_shot_incident("burn") / (burn / area) - 1) < 1e-12


def test_He_conservation_two_shots():
    from ife_wall.he.simulate import FastSimulation
    from ife_wall.scenarios import converged_preshot
    cyc = ROOT / f"runs/he_retention/thermal/Prototype_T{converged_preshot('Prototype', legacy=609.3):.1f}_ss.npz"
    sim = FastSimulation("Prototype", C.baseline(), str(cyc))
    for _ in range(2):
        sim.shot()
    sn = sim.snapshot()
    assert abs(sn["He_cons_err"]) / sn["He_in"] < 1e-10


def test_neutron_dose_scaled_to_the_scenario():
    """The per-shot neutron dose (154 MJ at 6.5 m) scales with the neutron fluence at the wall; Demo keeps it exactly."""
    from ife_wall.neutron import fluence_scale
    from ife_wall.scenarios import SCENARIOS
    assert fluence_scale(SCENARIOS["Demo"]) == 1.0
    assert abs(fluence_scale(SCENARIOS["Prototype"]) - 50 / 154) < 1e-15
    assert abs(fluence_scale(SCENARIOS["Experimental"]) - 20 / 154 * (6.5 / 5.0) ** 2) < 1e-15
