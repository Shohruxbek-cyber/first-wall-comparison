"""Post-processing of He-retention runs: loading, shots to thresholds (computed or extrapolated), derived numbers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "he_retention"
THRESHOLDS = {"damage_low_1e18": 1e18, "damage_high_1e19": 1e19, "blister_low_1e21": 1e21, "blister_high_1e22": 1e22}
REP_HZ = {"Experimental": None, "Prototype": 1.0, "Demo": 10.0}
SCEN_COLOR = {"Experimental": "#1b9e77", "Prototype": "#d95f02", "Demo": "#7570b3"}


def load(path):
    path = Path(path)
    d = np.genfromtxt(path / "totals.csv", delimiter=",", names=True)
    out = {c: np.atleast_1d(d[c]) for c in d.dtype.names}
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8")) if (path / "meta.json").exists() else {}
    prof = {}
    if (path / "profiles.npz").exists():
        z = np.load(path / "profiles.npz", allow_pickle=True)
        for k in z.files:
            shot, name = k.split("_", 1)
            prof.setdefault(int(shot), {})[name] = z[k]
    elif (path / "checkpoint.npz").exists():
        z = np.load(path / "checkpoint.npz", allow_pickle=True)
        for shot, d in z["profs"].item().items():
            prof[int(shot)] = d
    return dict(t=out, meta=meta, prof=prof, path=str(path))


def increments(shots, R, window):
    """mean retained-He increment per shot over sliding windows"""
    n = len(shots)
    idx = np.arange(window, n)
    return shots[idx], (R[idx] - R[idx - window]) / (shots[idx] - shots[idx - window])


def shots_to(threshold, shots, R):
    """Number of shots at which the retained He first reaches `threshold`.
    Returns dict(value, kind, lower, upper): kind 'computed' if reached inside the run (linear interpolation between
    shots), else 'extrapolated' with two bounds from the late-time increment per shot inc(N):
      lower = continue with the last increment (valid lower bound if inc decreases with N),
      upper = inc(N) = inc_inf + A N^-p fitted to the run and integrated (if the fit is reliable), else None."""
    shots, R = np.asarray(shots, float), np.asarray(R, float)
    if R.max() >= threshold:
        i = int(np.argmax(R >= threshold))
        if i == 0:
            v = shots[0] * threshold / R[0]
        else:
            v = shots[i - 1] + (threshold - R[i - 1]) / (R[i] - R[i - 1]) * (shots[i] - shots[i - 1])
        return dict(value=float(v), kind="computed", lower=float(v), upper=float(v))
    N = shots[-1]
    w = max(25, int(0.1 * len(shots)))
    inc_late = (R[-1] - R[-1 - w]) / (shots[-1] - shots[-1 - w])
    lower = N + (threshold - R[-1]) / inc_late
    upper = None
    try:
        ns, inc = increments(shots, R, w)
        sel = ns >= min(100, 0.2 * shots[-1])
        f = lambda n, a, A, p: a + A * n ** (-p)
        popt, _ = curve_fit(f, ns[sel], inc[sel], p0=[inc[-1], inc[sel][0] - inc[-1], 0.5],
                            bounds=([0, 0, 0.05], [10 * inc.max(), 1e3 * inc.max(), 3.0]), maxfev=20000)
        res = np.abs(f(ns[sel], *popt) - inc[sel]).max() / inc[sel].mean()
        if res < 0.03:
            # integrate R(N) = R_last + int_N^Nt inc dn  (numerically, log-spaced)
            nn = np.geomspace(N, 1e9, 20000)
            incf = f(nn, *popt)
            cum = R[-1] + np.r_[0.0, np.cumsum(0.5 * (incf[1:] + incf[:-1]) * np.diff(nn))]
            if cum[-1] >= threshold:
                upper = float(np.interp(threshold, cum, nn))
            else:
                upper = float("inf")
            fit = dict(inc_inf=float(popt[0]), A=float(popt[1]), p=float(popt[2]), max_rel_residual=float(res))
        else:
            fit = None
    except Exception:
        fit = None
    return dict(value=float(lower), kind="extrapolated", lower=float(lower), upper=upper, inc_late=float(inc_late), fit=fit)


def time_s(n_shots, scenario):
    f = REP_HZ[scenario]
    return None if f is None or n_shots is None or not np.isfinite(n_shots) else n_shots / f


def fmt_time(sec):
    if sec is None:
        return "n/a"
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    if sec < 2 * 86400:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} d"
