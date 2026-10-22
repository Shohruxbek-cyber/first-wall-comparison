"""Ion spectra (revision of 22 Oct 2026): the conserving construction keeps the number and
the energy of every species; the earlier constructions ("normalized", "raw") are reproduced bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import ions  # noqa: E402
from ife_wall.ions import ACCOUNTING_MJ, KEV_J, MEV, N_FUSIONS_REF, IonBin, bin_shape, load_ion_bins  # noqa: E402
from ife_wall.scenarios import Options  # noqa: E402


def _frozen_load_ion_bins(normalize=True):
    """ife_wall.ions.load_ion_bins as committed before 22 Oct 2026 (commit 1e3effe), verbatim."""
    rows = [r.split(",") for r in (ions.DATA / "aries_nrl154_dd_spectra.csv").read_text().splitlines()[2:] if r]
    bins = []
    for (g, s), E_ref in ACCOUNTING_MJ.items():
        d = np.array([(float(r[2]), float(r[3])) for r in rows if r[0] == g and r[1] == s])
        E, n = d[:, 0], d[:, 1]
        keep = n > 1.0
        E, n = E[keep], n[keep]
        lE = np.log(E)
        mid = np.exp(0.5 * (lE[1:] + lE[:-1]))
        lo = np.r_[E[0] ** 2 / mid[0], mid]
        hi = np.r_[mid, E[-1] ** 2 / mid[-1]]
        num = n * (hi - lo)
        en = num * E * KEV_J
        scale = E_ref * 1e6 / en.sum() if normalize else 1.0
        lnE_tab, lnn_tab = np.log(E * KEV_J), np.log(n)
        for a, b, e, N, Eb in zip(lo, hi, E, num, en):
            bins.append(IonBin(g, s, a * KEV_J, b * KEV_J, e * KEV_J, Eb * scale, N * scale, lnE_tab, lnn_tab))
    return bins


def _frozen_bin_shape(b, E, n_quad=400):
    """ife_wall.ions.bin_shape as committed before 22 Oct 2026, verbatim (without the cache)."""
    from scipy.optimize import brentq
    x = np.geomspace(b.E_lo, b.E_hi, n_quad)
    base = np.exp(np.interp(np.log(x), b.lnE_tab, b.lnn_tab))
    target = b.energy / b.number

    def mean_E(c):
        w = base * np.exp(c * np.log(x / b.E_mean))
        return np.trapezoid(w * x, x) / np.trapezoid(w, x)

    c = brentq(lambda c: mean_E(c) - target, -200.0, 200.0, xtol=1e-12)
    w = base * np.exp(c * np.log(x / b.E_mean))
    A = b.number / np.trapezoid(w, x)
    E = np.asarray(E, float)
    return A * np.exp(np.interp(np.log(E), b.lnE_tab, b.lnn_tab)) * np.exp(c * np.log(E / b.E_mean))


FIELDS = ("group", "species", "E_lo", "E_hi", "E_mean", "energy", "number")


@pytest.mark.parametrize("spectra,normalize", [("normalized", True), ("raw", False)])
def test_1_earlier_spectra_bit_for_bit(spectra, normalize):
    old, new = _frozen_load_ion_bins(normalize), load_ion_bins(spectra=spectra)
    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert all(getattr(a, f) == getattr(b, f) for f in FIELDS)
        assert np.array_equal(a.lnE_tab, b.lnE_tab) and np.array_equal(a.lnn_tab, b.lnn_tab)
        E = np.geomspace(a.E_lo, a.E_hi, 31)
        assert np.array_equal(_frozen_bin_shape(a, E), bin_shape(b, E))
    if not normalize:  # the earlier switch still selects the raw tables
        assert all(a.number == b.number for a, b in zip(old, load_ion_bins(False)))


def test_2_conserving_number_and_energy():
    t = ions.ion_spectra_tables("conserving")
    raw = ions.ion_spectra_tables("raw")
    for (g, s), d in t.items():
        assert d["energy"].sum() == pytest.approx(ACCOUNTING_MJ[(g, s)] * 1e6, rel=1e-12)
        if s != "4He":
            assert d["number"].sum() == pytest.approx(d["num_table"].sum(), rel=1e-12)
            assert d["num_table"].sum() == pytest.approx(raw[(g, s)]["num_table"].sum(), rel=1e-12)
    he4 = t[("burn", "4He")]["number"].sum() + t[("debris", "4He")]["number"].sum()
    assert he4 == pytest.approx(N_FUSIONS_REF, rel=1e-12)
    assert N_FUSIONS_REF == pytest.approx(154e6 / (17.6 * MEV), rel=1e-15)
    # the burn/debris split of 4He is the table's
    r_tab = t[("burn", "4He")]["num_table"].sum() / t[("debris", "4He")]["num_table"].sum()
    assert t[("burn", "4He")]["number"].sum() / t[("debris", "4He")]["number"].sum() == pytest.approx(r_tab, rel=1e-12)


def test_3_conserving_bins():
    bins = load_ion_bins(spectra="conserving")
    b4 = [b for b in bins if b.group == "burn" and b.species == "4He"]
    assert np.allclose([(b.E_hi - b.E_lo) / KEV_J for b in b4[1:8]], 500.0, rtol=1e-12)
    assert b4[0].E_hi / KEV_J == pytest.approx(500.0) and b4[8].E_lo / KEV_J == pytest.approx(4000.0)
    for b in bins:
        assert b.E_lo < b.E_mean < b.E_hi and b.number > 0 and b.energy > 0
        E = np.geomspace(b.E_lo, b.E_hi, 4001)
        w = bin_shape(b, E)
        assert np.all(np.isfinite(w)) and np.all(w >= 0)
        assert np.trapezoid(w, E) == pytest.approx(b.number, rel=2e-3)
        assert np.trapezoid(w * E, E) / np.trapezoid(w, E) == pytest.approx(b.E_mean, rel=2e-3)
    # burn ions keep their table energies; debris ions are stretched in energy
    t = ions.ion_spectra_tables("conserving")
    for b in bins:
        d = t[(b.group, b.species)]
        assert (d["stretch"] == 1.0) == (b.group == "burn")
        assert np.min(np.abs(d["E"] * d["stretch"] * KEV_J / b.E_mean - 1)) < 1e-12


def test_4_default_and_options():
    assert ions.DEFAULT_ION_SPECTRA in ions.ION_SPECTRA
    assert Options().ion_spectra == ions.DEFAULT_ION_SPECTRA
    with pytest.raises(ValueError):
        load_ion_bins(spectra="normalised")
