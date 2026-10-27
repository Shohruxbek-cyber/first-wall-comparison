"""Regenerate every figure of the research article (docs/article/*.md) from the stored run outputs.

Journal format:
  * size set in mm: Figs. 4 and 7 and every figure with three or more panels 190 mm wide, the
    others 90 mm (single column); height as needed; no rescaling afterwards;
  * every text at least 8 pt at that size (super- and subscripts at the usual 70 %);
  * no titles; panel letters (a), (b), ... ; American spelling; units in parentheses with real
    superscripts and minus signs; the symbols of the article: phi (conductivity factor), d (layer
    thickness), delta (film thickness), f_rr, s, tau_rr, f, h, T_Li;
  * Okabe-Ito colors, with line styles and markers where curves could be confused;
  * reference lines and bands of the earlier figures kept and labeled inside the plot;
  * PDF with embedded TrueType (Type 42) fonts plus a 600 dpi PNG preview: fig1 ... fig8, figC1,
    figD1, figE1, figH1, figH2, figH3.

Only stored outputs are read; no simulation is run.  Two figures differ:
  * Fig. D.1 recomputes the depth-model curves (projected and CSDA) from the stored SRIM tables
    and TRIM runs (data/srim) with the package's own functions, exactly as
    validation/srim_checks.py does; this takes a few seconds and runs no solver;
  * Fig. C.1 needs solver curves that no stored file contains (scripts/plot_verification.py runs
    the solver on the fly; runs/verification/results.json holds only error summaries).  It is
    drawn when runs/verification/verification_curves.npz exists and skipped otherwise.  That file
    can be written by plot_verification.py with np.savez(out / "verification_curves.npz", **curves)
    and these arrays (names as in plot_verification.py):
        a_t_s = t, a_dT_K = Ts - 300, a_t_exact_s = tt, a_dT_exact_K = 2*q/K*np.sqrt(ALPHA*tt/np.pi)
        b_times_s = (1e-10, 1e-9, 1e-8, 1e-7); for the i-th time: b_x_um_<i> = x*1e6,
        b_dT_K_<i> = s.T - 300, b_dT_exact_K_<i> = beer_exact(x, tk - tau/2, F, mu)
        c_t_s = t[::20], c_X_mm = X[::20]*1e3, c_t_exact_s = t, c_X_exact_mm = 2*lam*np.sqrt(all_*t)*1e3

The script writes only into the output folder.  sys.dont_write_bytecode keeps src/ free of
__pycache__ when Fig. D.1 imports the package.  After the figures it prints a layout check (font
sizes, texts outside the figure, overlaps of labels, legends and data) and a comparison of the
plotted numbers with the numbers quoted in the article draft of 20 Oct 2026.

Usage:  python scripts/make_article_figures.py [--out=DIR] [--only=fig1,figD1,...]
Default output folder: docs/figures/article
"""
import sys

sys.dont_write_bytecode = True  # must precede the import of ife_wall (Fig. D.1)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator, MultipleLocator,  # noqa: E402
                               NullFormatter, ScalarFormatter)
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "figures" / "article"
VERIF_CURVES = ROOT / "runs" / "verification" / "verification_curves.npz"

MM = 1 / 25.4                     # inch per mm
W1, W2 = 90 * MM, 190 * MM        # single column, full page width
FS, FS_LAB, FS_PANEL = 8.0, 8.5, 9.0
T_MELT, T_LIMIT = 3695.0, 925.0
SCEN = ("Experimental", "Prototype", "Demo")
NOCHECK = "nocheck"               # gid of artists the overlap check ignores (break marks)
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
     "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#8C8C8C", "band": "#DCDCDC"}  # Okabe-Ito

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS_LAB, "axes.titlesize": FS_PANEL, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 3.0,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "legend.frameon": False, "legend.handlelength": 2.6,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.5, "legend.borderpad": 0.3, "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.4,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.6, "lines.markeredgewidth": 0.7,
    "errorbar.capsize": 2.0, "axes.unicode_minus": True, "axes.formatter.use_mathtext": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
    "figure.constrained_layout.h_pad": 2.0 / 72, "figure.constrained_layout.w_pad": 2.0 / 72,
    "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
})


class MissingData(Exception):
    """A figure whose data are not in any stored file."""


# ------------------------------------------------------------------------------------------ inputs
SOURCES, CHECKS, WARN = {}, [], {}


def _rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def note_src(fid, p):
    lst = SOURCES.setdefault(fid, [])
    if _rel(p) not in lst:
        lst.append(_rel(p))


def jload(fid, rel):
    p = ROOT / rel
    note_src(fid, p)
    return json.loads(p.read_text(encoding="utf-8"))


def cload(fid, rel):
    p = ROOT / rel
    note_src(fid, p)
    return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)


def crossing_delta(rows, limit=T_LIMIT, T="peak_substrate_K", ok="substrate_pass"):
    """As scripts/make_summary.py: film thickness where the peak steel temperature crosses the limit,
    log-interpolated between the last failing and the first passing thickness."""
    rows = sorted(rows, key=lambda r: r["delta_um"])
    for a, b in zip(rows[:-1], rows[1:]):
        if (not a[ok]) and b[ok]:
            Ta, Tb = a[T], b[T]
            if a["status"] != "ok" or Ta <= Tb:
                return b["delta_um"]
            f = (Ta - limit) / (Ta - Tb)
            return float(np.exp(np.log(a["delta_um"]) + f * np.log(b["delta_um"] / a["delta_um"])))
    return rows[0]["delta_um"] if rows and rows[0][ok] else None


# ------------------------------------------------------------------------------------------ checks
def _agrees(v, quoted):
    """True if v rounds to the quoted number at its printed precision ('3145', '1.12e-6', '-44')."""
    q = quoted.replace("−", "-").strip()
    if "e" in q.lower():
        mant, ex = q.lower().split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return abs(v / 10 ** int(ex) - float(mant)) <= 0.5 * 10 ** -dec * (1 + 1e-9)
    dec = len(q.split(".")[1]) if "." in q else 0
    return abs(v - float(q)) <= 0.5 * 10 ** -dec * (1 + 1e-9)


def check(fid, what, value, quoted):
    CHECKS.append((fid, what, f"{value:.6g}", quoted, _agrees(value, quoted)))


def _agrees_up(v, quoted):
    """True if the quoted number is v rounded UP at its printed precision (a minimum thickness, 27 Oct 2026)."""
    q = quoted.strip()
    dec = len(q.split(".")[1]) if "." in q else 0
    return abs(np.ceil(v * 10 ** dec - 1e-9) / 10 ** dec - float(q)) < 0.5 * 10 ** -dec


def check_min(fid, what, value, quoted):
    CHECKS.append((fid, what, f"{value:.6g}", quoted, _agrees_up(value, quoted)))


def check_true(fid, what, cond, shown=""):
    CHECKS.append((fid, what, shown, "", bool(cond)))


# ------------------------------------------------------------------------------------------ layout helpers
def letter(ax, s):
    ax.set_title(f"({s})", loc="left", fontsize=FS_PANEL, fontweight="bold", pad=3.0)


def plain_log(axis):
    """Log axis with plain decimal labels (0.01, 0.1, 1, 10, ...) and unlabeled minor ticks."""
    axis.set_major_locator(LogLocator(base=10, numticks=12))
    axis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


class NoZeroFormatter(ScalarFormatter):
    """Default tick labels without the label at zero (where x and y axes meet in a corner)."""

    def __call__(self, x, pos=None):
        return "" if abs(x) < 1e-12 else super().__call__(x, pos)


def sci_tex(x, digits=2):
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.{digits}f}$\times$10$^{{{e}}}$"


def fmt_um(d):
    return f"{d:.3g}" if d < 1000 else f"{d:.0f}"


def _box(bb, pad):
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _overlap(a, b, tol=0.5):
    return min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol


def _inside(a, b, tol=1.0):
    return a.x0 >= b.x0 - tol and a.x1 <= b.x1 + tol and a.y0 >= b.y0 - tol and a.y1 <= b.y1 + tol


def _hits_data(ax, bb, r, pad_pt=0.8):
    """Description of the first line, marker, bar or collection of `ax` that touches the box bb."""
    px = ax.figure.dpi / 72.0
    box = _box(bb, pad_pt * px)
    for ln in ax.get_lines():
        if not ln.get_visible() or ln.get_gid() == NOCHECK:
            continue
        mk = ln.get_marker() not in (None, "", " ", "None", "none")
        b2 = _box(box, ln.get_markersize() / 2 * px) if mk else box
        if ln.get_linestyle() in ("None", "none", "", " "):
            xy = ln.get_transform().transform(ln.get_xydata())
            if np.any((xy[:, 0] >= b2.x0) & (xy[:, 0] <= b2.x1) & (xy[:, 1] >= b2.y0) & (xy[:, 1] <= b2.y1)):
                return f"markers {ln.get_label()!r}"
        elif ln.get_transform().transform_path(ln.get_path()).intersects_bbox(b2, filled=False):
            return f"line {ln.get_label()!r}"
    for p in ax.patches:
        if p.get_visible() and p.get_gid() not in (NOCHECK, "band") and _overlap(box, p.get_window_extent(r)):
            return "bar"
    for c in ax.collections:
        if not c.get_visible() or c.get_gid() == NOCHECK:
            continue
        tr = c.get_transform()
        if any(tr.transform_path(pth).intersects_bbox(box, filled=False) for pth in c.get_paths()):
            return "error bar"
    return None


def place(ax, options, fallback=None, draw=True, **kw):
    """Add a label at the first free option.  Each option is (text, x, y, ha, va, coords, dy_pt) with
    coords 'axes' (x, y in axes fractions) or 'yline' (x in axes fraction, y in data units); free means
    inside the axes and touching no data artist, label or legend of the axes.  Returns the text, or,
    when no option is free, the first option (fallback='first') or None.  Call it after everything
    else of the figure exists: the final layout decides what is free."""
    fig = ax.figure
    if draw:
        fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    taken = [t.get_window_extent(r) for t in ax.texts if t.get_visible()]
    if ax.get_legend() is not None:
        taken.append(ax.get_legend().get_window_extent(r))
    kw.setdefault("fontsize", FS)
    for s, x, y, ha, va, coords, dy in options:
        tr = ax.transAxes if coords == "axes" else ax.get_yaxis_transform()
        t = ax.annotate(s, (x, y), xycoords=tr, xytext=(0, dy), textcoords="offset points", ha=ha, va=va, **kw)
        bb = t.get_window_extent(r)
        if _inside(bb, axbb) and _hits_data(ax, bb, r) is None and not any(_overlap(bb, o) for o in taken):
            return t
        t.remove()
    if fallback == "first":
        s, x, y, ha, va, coords, dy = options[0]
        tr = ax.transAxes if coords == "axes" else ax.get_yaxis_transform()
        return ax.annotate(s, (x, y), xycoords=tr, xytext=(0, dy), textcoords="offset points", ha=ha, va=va, **kw)
    return None


def hline_options(y, texts, xs=(0.02, 0.98, 0.5, 0.3, 0.7)):
    """Label options for a horizontal reference line: above and below it, at the given x positions."""
    out = []
    for s in texts:
        for x in xs:
            ha = "left" if x < 0.4 else ("right" if x > 0.6 else "center")
            out += [(s, x, y, ha, "bottom", "yline", 1.6), (s, x, y, ha, "top", "yline", -1.6)]
    return out


def ref_line(ax, y, ls="--", color="k"):
    ax.axhline(y, color=color, ls=ls, lw=0.8, zorder=1.5, label="_ref")


def label_line(fid, axes, y, variants, xs=(0.02, 0.98, 0.5, 0.3, 0.7), draw=True):
    """Label the reference line at y in every axes of `axes` with the first text variant that finds a
    free place in all of them.  Returns False (and records a warning) if none does."""
    if draw:
        axes[0].figure.canvas.draw()  # final layout; labels inside the axes do not change it
    for s in variants:
        placed = [place(ax, hline_options(y, [s], xs), draw=False) for ax in axes]
        if all(t is not None for t in placed):
            return True
        for t in placed:
            if t is not None:
                t.remove()
    WARN.setdefault(fid, []).append(f"no free place for the label of the line at {y:g}")
    return False


def tornado(ax, labels, vals, annot, order=None):
    """Horizontal bars sorted by magnitude (largest on top), as in scripts/make_summary.py, or in a given
    order; increase in vermillion, decrease in blue; the value of each bar printed at its end."""
    if order is None:
        order = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
    n = len(order)
    y = np.arange(n)[::-1]
    v = np.array([vals[i] for i in order])
    ax.barh(y, v, height=0.62, color=[C["vermillion"] if x > 0 else C["blue"] for x in v], lw=0, zorder=2)
    ax.axvline(0, color="k", lw=0.6, zorder=3, label="_zero")
    ax.set_yticks(y, [labels[i] for i in order])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_ylim(-0.7, n - 0.3)
    lo, hi = min(0.0, v.min()), max(0.0, v.max())
    span = hi - lo
    ax.set_xlim(lo - (0.30 if lo < 0 else 0.06) * span, hi + (0.30 if hi > 0 else 0.06) * span)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    for yy, x, i in zip(y, v, order):
        right = x >= -2e-3 * span  # bars too short to see are annotated on the right of zero
        ax.annotate(annot(i), (x, yy), xytext=(2.5 if right else -2.5, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=FS)


# ------------------------------------------------------------------------------------------ Fig. 1
STY_SCEN = {"Experimental": (C["blue"], "-", "o"), "Prototype": (C["orange"], "--", "s"),
            "Demo": (C["vermillion"], "-.", "^")}


def fig1():
    F = "fig1"
    fig, ax = plt.subplots(figsize=(W1, 64 * MM), layout="constrained")
    ax.axhspan(1300, 1800, color=C["band"], lw=0, zorder=0).set_gid("band")
    quoted = {"Experimental": ("868", "1.65", "300"), "Prototype": ("1371", "2.14", "600"), "Demo": ("3027", "2.15", "600")}
    for s in SCEN:
        h = cload(F, f"runs/w_single/baseline/{s}_history.csv")
        t_us, T = h[:, 0] * 1e6, h[:, 1]
        col, ls, _ = STY_SCEN[s]
        ax.plot(t_us, T, color=col, ls=ls, lw=1.1, label=s)
        i = int(np.argmax(T))
        check(F, f"{s}: peak surface temperature (K)", T[i], quoted[s][0])
        check(F, f"{s}: time of the peak (µs)", t_us[i], quoted[s][1])
        check(F, f"{s}: initial temperature (K)", T[0], quoted[s][2])
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 50)
    ax.set_ylim(200, 4000)
    plain_log(ax.xaxis)
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(250))
    ax.set_xlabel("Time after ignition (µs)")
    ax.set_ylabel("Surface temperature (K)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ref_line(ax, T_MELT)
    label_line(F, [ax], T_MELT, ["Melting point 3695 K", "Melting point\n3695 K"], xs=(0.98, 0.02, 0.5))
    band = []
    for s in ("Recrystallization range 1300–1800 K", "Recrystallization\n1300–1800 K"):
        for x, ha in ((0.02, "left"), (0.98, "right")):
            band += [(s, x, 1550, ha, "center", "yline", 0), (s, x, 1800, ha, "bottom", "yline", 1.6),
                     (s, x, 1300, ha, "top", "yline", -1.6)]
    if place(ax, band, draw=False) is None:
        WARN.setdefault(F, []).append("no free place for the recrystallization label")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 2
def broken_scale(lo, hi, gap):
    """Piecewise-linear scale: [.., lo] and [hi, ..] at the same K per mm, (lo, hi) squeezed into gap units."""
    k = gap / (hi - lo)

    def fwd(y):
        y = np.asarray(y, float)
        return np.where(y <= lo, y, np.where(y >= hi, y - (hi - lo) + gap, lo + (y - lo) * k))

    def inv(u):
        u = np.asarray(u, float)
        return np.where(u <= lo, u, np.where(u >= lo + gap, u + (hi - lo) - gap, lo + (u - lo) / k))
    return fwd, inv


def mark_break(ax, lo, hi):
    tr = ax.get_yaxis_transform()
    for x in (0.0, 1.0):
        ax.add_patch(Rectangle((x - 0.02, lo), 0.04, hi - lo, transform=tr, facecolor="white", edgecolor="none",
                               clip_on=False, zorder=4, gid=NOCHECK))
    ax.plot([0, 0, 1, 1], [lo, hi, lo, hi], transform=tr, marker=[(-1, -0.55), (1, 0.55)], markersize=5,
            ls="none", color="k", mec="k", mew=0.7, clip_on=False, zorder=5, gid=NOCHECK, label="_break")


def fig2():
    F = "fig2"
    a = cload(F, "runs/w_multi/baseline/Demo_per_shot.csv")
    j = jload(F, "runs/w_multi/baseline/Demo.json")
    n, pre, peak, itf, back = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(W1, 112 * MM), layout="constrained", sharex=True,
                                   height_ratios=[1.0, 1.35])
    axa.plot(n, pre, color=C["blue"], ls="-", label="Surface")
    axa.plot(n, back, color=C["vermillion"], ls=":", lw=1.4, label="Steel back face")
    axa.set_ylim(580, 950)
    axa.yaxis.set_minor_locator(MultipleLocator(25))
    axa.set_ylabel("Pre-shot temperature (K)")
    axa.legend(loc="lower right")
    letter(axa, "a")
    lo, hi, gap = 965.0, 2980.0, 28.0  # the upper part starts below the first-shot peak (3027 K)
    axb.set_yscale("function", functions=broken_scale(lo, hi, gap))
    axb.plot(n, peak, color=C["blue"], ls="-", label="Surface")
    axb.plot(n, itf, color=C["green"], ls="-.", label="W/steel interface")
    ref_line(axb, T_LIMIT)
    axb.set_ylim(580, 3480)
    axb.yaxis.set_major_locator(FixedLocator([600, 700, 800, 900, 3000, 3200, 3400]))
    axb.yaxis.set_minor_locator(FixedLocator([650, 750, 850, 950, 3100, 3300]))
    axb.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    axb.yaxis.set_minor_formatter(NullFormatter())
    mark_break(axb, lo, hi)
    axb.set_xlim(0, 605)
    axb.xaxis.set_major_locator(MultipleLocator(100))
    axb.xaxis.set_minor_locator(MultipleLocator(50))
    axb.set_xlabel("Shot number")
    axb.set_ylabel("Peak temperature in the shot (K)")
    axb.legend(loc="lower right")
    letter(axb, "b")
    label_line(F, [axb], T_LIMIT, ["Steel limit 925 K"], xs=(0.3, 0.5, 0.7, 0.98))  # clear of the break mark
    check(F, "shots in the run", n[-1], "600")
    check(F, "pre-shot surface, shot 600 (K)", pre[-1], "921")
    check(F, "pre-shot surface, shot 600 (K), Fig. 4 caption", pre[-1], "921")
    check(F, "pre-shot rise over the last 60 shots (K/shot)", (pre[-1] - pre[-60]) / 59, "0.035")
    check(F, "peak surface, last shot (K)", peak[-1], "3317")
    check(F, "peak interface, shot 600 (K)", itf[-1], "934")
    check(F, "first shot with the interface above 925 K", n[np.argmax(itf > T_LIMIT)], "433")
    check(F, "peak surface, first shot (K)", peak[0], "3027")
    check(F, "pre-shot rise in the JSON summary (K/shot)", j["preshot_rise_per_shot_last10pct_K"], "0.035")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 3
# labels "baseline -> changed value" (presentation review, 27 Oct 2026); the baselines are the RunOptions defaults of
# src/ife_wall/scenarios.py (checked against the baseline run below): conserving spectra, k_scale 1, projected
# depth, xray_tau 20 ns, alpha_evap 1, w_lv "raffray_table" = 4009 kJ/kg (src/ife_wall/library.py, W_LV)
W_CASES = [("normalize_ions-false", "ion spectra: production → raw tables"), ("k_scale-0.85", r"$k$: ×1 → ×0.85"),
           ("k_scale-1.15", r"$k$: ×1 → ×1.15"), ("depth_mode-csda", "depth: projected → CSDA"),
           ("xray_tau-1e-10", "x-ray pulse: 20 → 0.1 ns"), ("xray_tau-1e-08", "x-ray pulse: 20 → 10 ns"),
           ("xray_tau-5e-08", "x-ray pulse: 20 → 50 ns"), ("alpha_evap-0.1", r"$\alpha$: 1 → 0.1"),
           ("w_lv-raffray_slope", r"$L_\mathrm{v}$: 4009 → 4630 kJ kg$^{-1}$")]
W_BASE = {"xray_tau": 2e-08, "k_scale": 1.0, "depth_mode": "projected", "alpha_evap": 1.0, "w_lv": "raffray_table",
          "normalize_ions": True, "ion_spectra": "conserving"}
# left out: runs/w_single/stretch-false ("no H-isotope depth stretch", a null case)


def _ratio_txt(r):
    return f"×{r:.0f}" if r >= 10 else f"×{r:.2f}"


def _dT_txt(v):
    return f"{v:+.0f}".replace("-", "−") if abs(v) >= 1 else (f"{v:+.1f}".replace("-", "−") if abs(v) >= 0.05 else "0.0")


def fig3():
    F = "fig3"
    b = jload(F, "runs/w_single/baseline/Demo.json")
    T0, A0 = b["peak_surface_T_K"], b["ablated_nm_per_shot"]
    check_true(F, "baseline options as the labels state", all(b["options"][k] == v for k, v in W_BASE.items()),
               str({k: b["options"][k] for k in W_BASE}))
    labs, dT, lr, ratio = [], [], [], []
    for tag, lab in W_CASES:
        j = jload(F, f"runs/w_single/{tag}/Demo.json")
        changed = {k: j["options"][k] for k in W_BASE if j["options"][k] != b["options"][k]}
        check_true(F, f"{tag}: exactly one input changed", len(changed) == 1, str(changed))
        labs.append(lab)
        dT.append(j["peak_surface_T_K"] - T0)
        ratio.append(j["ablated_nm_per_shot"] / A0)
        lr.append(np.log10(ratio[-1]))
    order = sorted(range(len(dT)), key=lambda i: abs(dT[i]), reverse=True)  # (b) keeps the order of (a)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 112 * MM), layout="constrained")
    tornado(a1, labs, dT, lambda i: _dT_txt(dT[i]), order)
    a1.set_xlabel(f"Peak surface temperature change (K)\nfrom the baseline of {T0:.0f} K")
    letter(a1, "a")
    tornado(a2, labs, lr, lambda i: _ratio_txt(ratio[i]), order)
    a2.set_xlabel("log$_{10}$ of the evaporated-depth ratio\nto the baseline of " + sci_tex(A0) + " nm")
    letter(a2, "b")
    d = dict(zip([t for t, _ in W_CASES], dT))
    check(F, "baseline peak (K)", T0, "3027")
    check(F, "baseline evaporated depth (nm)", A0, "3.38e-7")
    check(F, "raw ion tables (K)", d["normalize_ions-false"], "526")
    check(F, "conductivity ×0.85 (K)", d["k_scale-0.85"], "180")
    check(F, "conductivity ×1.15 (K)", d["k_scale-1.15"], "-148")
    check(F, "CSDA depth model (K)", d["depth_mode-csda"], "-40")
    r = dict(zip([t for t, _ in W_CASES], ratio))
    check(F, "raw ion tables, evaporated-depth ratio (Section 4.3)", r["normalize_ions-false"], "133")
    check(F, "conductivity ×0.85, evaporated-depth ratio (Section 4.3)", r["k_scale-0.85"], "6.6")
    check(F, "evaporation coefficient 0.1, evaporated-depth ratio (Section 4.3)", r["alpha_evap-0.1"], "0.10")
    small = [abs(d[t]) for t in ("xray_tau-1e-10", "xray_tau-1e-08", "xray_tau-5e-08", "alpha_evap-0.1", "w_lv-raffray_slope")]
    check_true(F, "x-ray pulse, evaporation coefficient, heat of vaporization: |change| < 1 K", max(small) < 1, f"max {max(small):.2f} K")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 4
D_UM, PHI = (1.0, 3.0, 10.0), (1.0, 0.75, 0.5, 0.25)
STY_D = {1.0: (C["blue"], "o", "-"), 3.0: (C["orange"], "s", "--"), 10.0: (C["vermillion"], "^", "-.")}


def skin_json(start, scen, d, phi):
    if d == 0:
        return (f"results/skin_sweep/regression_check/{scen}.json" if start == "cold"
                else "results/skin_sweep/single/Demo_hot/ref_noskin/Demo.json")
    return f"results/skin_sweep/single/{'cold' if start == 'cold' else 'Demo_hot'}/d{d:g}_f{phi:g}/{scen}.json"


def fig4():
    F = "fig4"
    panels = [("Experimental", "cold"), ("Prototype", "cold"), ("Demo", "cold"), ("Demo", "hot")]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 66 * MM), layout="constrained")
    res = {}
    for ax, (scen, start), L in zip(axs, panels, "abcd"):
        for d in D_UM:
            js = [jload(F, skin_json(start, scen, d, f)) for f in PHI]
            for f, j in zip(PHI, js):
                res[(scen, start, d, f)] = j
            col, mk, ls = STY_D[d]
            ax.plot(PHI, [j["peak_surface_T_K"] for j in js], color=col, marker=mk, ls=ls, ms=4,
                    label=f"$d$ = {d:g} µm")
        ref = jload(F, skin_json(start, scen, 0, 1.0))
        res[(scen, start, 0, 1.0)] = ref
        ax.plot([1.0], [ref["peak_surface_T_K"]], ls="none", marker="*", ms=8.5, color="k", mec="k", zorder=5,
                label="No layer")
        ax.set_xlim(1.07, 0.18)
        ax.set_xticks(PHI, ["1", "0.75", "0.5", "0.25"])
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        letter(ax, L)
    for ax in axs[2:]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, T_MELT + 0.12 * (hi - lo)))
        ref_line(ax, T_MELT)
    axs[0].set_ylabel("Peak surface temperature (K)")
    fig.supxlabel(r"Conductivity factor $\varphi$ of the surface layer", fontsize=FS_LAB)
    h, lab = axs[0].get_legend_handles_labels()
    leg = axs[0].legend(h, lab, loc="upper left")  # one legend, inside panel (a), as in Fig. 5 (presentation review, 27 Oct 2026)
    if not label_line(F, axs[2:], T_MELT, ["Melting point 3695 K", "Melting point\n3695 K", "Melting\n3695 K"],
                      xs=(0.03, 0.97, 0.5)):
        leg.remove()  # fall back to a legend entry for the melting point
        axs[0].legend(h + [Line2D([], [], color="k", ls="--", lw=0.8)], lab + ["Melting point 3695 K"], loc="upper left")
    # numbers quoted in Section 4.5 and in the caption
    P = lambda sc, st, d, f: res[(sc, st, d, f)]["peak_surface_T_K"]  # noqa: E731
    M = lambda sc, st, d, f: res[(sc, st, d, f)]["melt_depth_um"]  # noqa: E731
    check(F, "no layer, Experimental (K)", P("Experimental", "cold", 0, 1.0), "868")
    check(F, "no layer, Prototype (K)", P("Prototype", "cold", 0, 1.0), "1371")
    check(F, "no layer, Demo from 600 K (K)", P("Demo", "cold", 0, 1.0), "3027")
    check(F, "no layer, Demo from 920.9 K (K)", P("Demo", "hot", 0, 1.0), "3317")
    check(F, "hot start temperature (K)", res[("Demo", "hot", 0, 1.0)]["T_init_K"], "921")  # Fig. 4 caption and Section 4.2 print 921 K (presentation review, 27 Oct 2026)
    check(F, "Demo 600 K, d = 1 µm, phi = 0.25 (K)", P("Demo", "cold", 1.0, 0.25), "3562")
    check(F, "Demo 600 K, d = 3 µm, phi = 0.5 (K)", P("Demo", "cold", 3.0, 0.5), "3564")
    melts = [M("Demo", "cold", 3.0, 0.25), M("Demo", "cold", 10.0, 0.5), M("Demo", "cold", 10.0, 0.25)]
    check(F, "melt depth range, 3 µm/0.25 and 10 µm/<=0.5 from 600 K, low end (µm)", min(melts), "0.3")
    check(F, "same, high end (µm)", max(melts), "1.5")
    check(F, "Demo hot, d = 10 µm, phi = 0.75 (K)", P("Demo", "hot", 10.0, 0.75), "3622")
    check(F, "Demo hot, d = 10 µm, phi = 0.25 (K)", P("Demo", "hot", 10.0, 0.25), "4102")
    check(F, "Demo hot, d = 10 µm, phi = 0.25, melt depth (µm)", M("Demo", "hot", 10.0, 0.25), "1.8")
    pre2 = [res[("Demo", "cold", d, f)]["pre_shot2_surface_T_K"] for d in D_UM for f in PHI]
    check_true(F, "pre-shot temperature of the second Demo shot is 610 K for every d and phi",
               all(_agrees(v, "610") for v in pre2), f"{min(pre2):.3f}-{max(pre2):.3f} K")
    check(F, "Prototype d = 1 µm, phi = 0.25 (K)", P("Prototype", "cold", 1.0, 0.25), "1522")
    above = {(d, f) for d in D_UM for f in PHI[1:] if P("Prototype", "cold", d, f) > 1523.0}
    check_true(F, "Prototype above 1523 K exactly for phi<=0.5 at d>=3",
               above == {(3.0, 0.5), (3.0, 0.25), (10.0, 0.5), (10.0, 0.25)},
               str(sorted(above)))
    return fig


# ------------------------------------------------------------------------------------------ Fig. 5
def fig5():
    F = "fig5"
    film = jload(F, "runs/deposition/li_film_absorption.json")
    d = np.array([r["delta_um"] for r in film])
    fig, ax = plt.subplots(figsize=(W1, 62 * MM), layout="constrained")
    for key, lab, col, mk, ls in (("threat_in_Li", "All threat energy", C["black"], "o", "-"),
                                  ("ion_in_Li", "Ions only", C["blue"], "s", "--"),
                                  ("xray_in_Li", "X-rays only", C["orange"], "^", "-.")):
        ax.plot(d, [r[key] for r in film], color=col, marker=mk, ls=ls, ms=3.4, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1250)
    ax.set_ylim(0, 1.0)
    plain_log(ax.xaxis)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(r"Lithium film thickness $\delta$ (µm)")
    ax.set_ylabel("Fraction absorbed in the film")
    ax.legend(loc="upper left")
    last = film[int(np.argmax(d))]
    check(F, "1 mm film: share of all threat energy (%)", 100 * last["threat_in_Li"], "91")
    check(F, "1 mm film: share of the ion energy (%)", 100 * last["ion_in_Li"], "96")
    check(F, "1 mm film: share of the x-ray energy (%)", 100 * last["xray_in_Li"], "9")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 6
def fig6():
    F = "fig6"
    rows = {s: sorted(jload(F, f"runs/li_scan/T600K/{s}.json")["rows"], key=lambda r: r["delta_um"]) for s in SCEN}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 64 * MM), layout="constrained")
    ref = rows["Demo"]
    same = all(len(rows[s]) == len(ref) and all(abs(a["delta_um"] - b["delta_um"]) < 1e-9 and
                                                 abs(a["absorbed_frac_in_Li"] - b["absorbed_frac_in_Li"]) < 1e-9
                                                 for a, b in zip(rows[s], ref)) for s in SCEN)
    if same:
        axs[0].plot([r["delta_um"] for r in ref], [r["absorbed_frac_in_Li"] for r in ref], color="k", marker="o",
                    ms=3.4, label="all three scenarios")
    for s in SCEN:
        col, ls, mk = STY_SCEN[s]
        ok = [r for r in rows[s] if r["status"] == "ok"]
        dd = [r["delta_um"] for r in ok]
        if not same:
            axs[0].plot([r["delta_um"] for r in rows[s]], [r["absorbed_frac_in_Li"] for r in rows[s]], color=col,
                        ls=ls, marker=mk, ms=3.4, label=s)
        axs[1].plot(dd, [r["peak_substrate_K"] for r in ok], color=col, ls=ls, marker=mk, ms=3.4, label=s)
        axs[2].plot(dd, [r["vaporized_kg_m2"] for r in ok], color=col, ls=ls, marker=mk, ms=3.4, label=s)
    for ax, L in zip(axs, "abc"):
        ax.set_xscale("log")
        ax.set_xlim(0.8, 1250)
        plain_log(ax.xaxis)
        ax.set_xlabel(r"Lithium film thickness $\delta$ (µm)")
        letter(ax, L)
    axs[0].set_ylim(0, 1.0)
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0].set_ylabel("Absorbed fraction of the threat energy")
    axs[1].set_ylim(560, 1850)  # the Demo 5.6 µm film peaks at 1791 K
    axs[1].yaxis.set_minor_locator(MultipleLocator(50))
    axs[1].set_ylabel("Peak steel temperature (K)")
    axs[1].legend(loc="upper right")
    ref_line(axs[1], T_LIMIT)
    axs[2].set_yscale("log")
    axs[2].set_ylim(2e-7, 3e-3)
    axs[2].yaxis.set_minor_formatter(NullFormatter())
    axs[2].set_ylabel(r"Evaporated mass per shot (kg m$^{-2}$)")
    label_line(F, [axs[1]], T_LIMIT, ["Steel limit 925 K"], xs=(0.98, 0.02, 0.5))
    if same:
        s = "Same for the three scenarios"
        place(axs[0], [(s, 0.97, 0.04, "right", "bottom", "axes", 0), (s, 0.03, 0.96, "left", "top", "axes", 0)],
              fallback="first", draw=False)
    D = {r["delta_um"]: r for r in rows["Demo"]}
    k = lambda x: min(D, key=lambda dd: abs(np.log(dd / x)))  # noqa: E731
    check(F, "Demo, 17.8 µm: peak steel (K)", D[k(17.8)]["peak_substrate_K"], "1116")
    check(F, "Demo, 31.6 µm: peak steel (K)", D[k(31.6)]["peak_substrate_K"], "907")
    check_true(F, "Demo films of 3.2 µm and less stopped (missing in b, c)",
               all(r["status"] != "ok" for r in rows["Demo"] if r["delta_um"] <= 3.2) and D[k(5.6)]["status"] == "ok")
    for s, q in (("Demo", "30.1"), ("Prototype", "11.9"), ("Experimental", "7.2")):
        check_min(F, f"{s}: minimum film for 925 K (µm)", crossing_delta(rows[s]), q)
    for s, q in (("Demo", "7.4e-4"), ("Prototype", "8.9e-6"), ("Experimental", "8.0e-7")):
        check(F, f"{s}: thick-film evaporation, median over >= 100 µm (kg m-2)",
              float(np.median([r["vaporized_kg_m2"] for r in rows[s] if r["delta_um"] >= 100])), q)
    check(F, "1 mm film: absorbed fraction (%)", 100 * D[k(1000)]["absorbed_frac_in_Li"], "91")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 7
STY_5 = [(C["vermillion"], "-", "o"), (C["orange"], "--", "s"), (C["green"], "-.", "^"), (C["blue"], ":", "D"),
         (C["purple"], (0, (6, 1.5, 1, 1.5, 1, 1.5)), "v")]


def fig7():
    F = "fig7"
    runs = []
    for p in sorted((ROOT / "runs" / "li_multi").glob("Demo_d*um")):
        if re.fullmatch(r"Demo_d[0-9.]+um", p.name) and (p / "Demo.json").exists():
            j = jload(F, _rel(p / "Demo.json"))
            runs.append((j["delta_um"], j, cload(F, _rel(p / "Demo_per_shot.csv"))))
    runs.sort(key=lambda t: t[0])
    fig, axs = plt.subplots(1, 3, figsize=(W2, 68 * MM), layout="constrained")
    for k, (d, j, r) in enumerate(runs):
        col, ls, mk = STY_5[k % len(STY_5)]
        kw = dict(color=col, ls=ls, lw=1.2, marker=mk, ms=3.6, markevery=(8 * k, 50), label=rf"$\delta$ = {fmt_um(d)} µm")
        axs[0].plot(r[:, 0], r[:, 1], **kw)
        axs[1].plot(r[:, 0], r[:, 2], **kw)
        axs[2].plot(r[:, 0], r[:, 5] * 1e4, **kw)
    for ax, L in zip(axs, "abc"):
        ax.set_xlim(0, 400)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.set_xlabel("Shot number")
        letter(ax, L)
    axs[0].set_ylabel("Pre-shot steel front face (K)")
    axs[0].yaxis.set_minor_locator(MultipleLocator(25))
    axs[1].set_ylabel("Peak steel temperature (K)")
    axs[1].yaxis.set_minor_locator(MultipleLocator(50))
    axs[2].set_ylabel(r"Evaporated mass per shot (10$^{-4}$ kg m$^{-2}$)")
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.1))
    ref_line(axs[1], T_LIMIT)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncols=len(h))
    label_line(F, [axs[1]], T_LIMIT, ["Steel limit 925 K"], xs=(0.98, 0.02, 0.5, 0.3, 0.7))
    R = {round(d, 1): (j, r) for d, j, r in runs}
    j31, r31 = R[31.6]
    check(F, "31.6 µm: first shot above 925 K", r31[np.argmax(r31[:, 2] > T_LIMIT), 0], "9")
    check(F, "31.6 µm: peak steel at the quasi-steady state (K)", r31[-1, 2], "1006")
    check(F, "31.6 µm: pre-shot front face, first shot (K)", r31[0, 1], "600")
    check(F, "31.6 µm: pre-shot front face, last shot (K)", r31[-1, 1], "783")
    check(F, "56.2 µm: peak steel, maximum over the run (K)", R[56.2][1][:, 2].max(), "871")
    ns = [len(r) for _, _, r in runs]
    check(F, "shortest run (shots)", min(ns), "100")
    check(F, "longest run (shots)", max(ns), "390")
    front = [r[-1, 1] for d, _, r in runs if 17 < d < 101]
    check(F, "pre-shot front face behind 17.8-100 µm, low end (K)", min(front), "728")
    check(F, "same, high end (K)", max(front), "795")
    check(F, "31.6 µm: rise of the evaporated mass per shot (%)", 100 * (r31[-1, 5] / r31[0, 5] - 1), "6")
    mrows = [{"delta_um": d, "peak_substrate_K": float(r[:, 2].max()), "status": j["status"],
              "substrate_pass": bool(r[:, 2].max() <= T_LIMIT and j["status"] == "ok")} for d, j, r in runs]
    check_min(F, "multi-shot minimum film (µm)", crossing_delta(mrows), "44.7")
    return fig


# ------------------------------------------------------------------------------------------ Fig. 8
VS_CASES = [((0.0, 1.0, 0.0), "frr0_s1"), ((0.5, 1.0, 0.0), "frr0.5_s1"), ((1.0, 1.0, 0.0), "frr1_s1"),
            ((0.0, 2.0, 0.0), "frr0_s2"), ((0.5, 2.0, 0.0), "frr0.5_s2"), ((1.0, 2.0, 0.0), "frr1_s2"),
            ((1.0, 1.0, 3e-4), "frr1_s1_tau300us")]
TABLE6 = {  # Table D.2 (Appendix D): thick-film evaporation (kg m-2), ratio, peak steel at 31.6 µm (K), minimum film (µm)
    "frr0_s1": ("5.87e-4", "0.79", "895", "29.1"), "frr0.5_s1": ("6.74e-4", "0.91", "898", "29.3"),
    "frr1_s1": ("7.78e-4", "1.05", "901", "29.5"), "frr0_s2": ("4.89e-4", "0.66", "888", "28.4"),
    "frr0.5_s2": ("6.23e-4", "0.84", "893", "28.8"), "frr1_s2": ("7.99e-4", "1.08", "896", "29.2"),
    "frr1_s1_tau300us": ("5.89e-4", "0.79", "896", "29.1")}


def fig8():
    F = "fig8"
    base = sorted(jload(F, "runs/li_scan/T600K/Demo.json")["rows"], key=lambda r: r["delta_um"])
    ok = [r for r in base if r["status"] == "ok"]
    dmin = min(r["delta_um"] for r in ok)  # every case from the thinnest film the baseline completes (as before)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 128 * MM), layout="constrained", sharex=True)
    kw0 = dict(color="k", ls="-", marker="o", ms=3.2, lw=1.1, label="No shielding", zorder=4)
    a1.plot([r["delta_um"] for r in ok], [r["vaporized_kg_m2"] * 1e4 for r in ok], **kw0)
    a2.plot([r["delta_um"] for r in ok], [r["peak_substrate_K"] for r in ok], **kw0)
    thick0 = float(np.median([r["vaporized_kg_m2"] for r in base if r["delta_um"] >= 100 and r["status"] == "ok"]))
    T31 = {"none": next(r for r in base if abs(r["delta_um"] - 31.6228) < 1e-3)["peak_substrate_K"]}
    for (frr, s, tau), tag in VS_CASES:
        rows_all = sorted(jload(F, f"results/vapor_shielding/li_Demo_{tag}.json")["rows"], key=lambda r: r["delta_um"])
        rows = [r for r in rows_all if r["status"] == "ok" and r["delta_um"] >= dmin * (1 - 1e-9)]
        col = C["green"] if tau else {0.0: C["blue"], 0.5: C["orange"], 1.0: C["vermillion"]}[frr]
        ls = ":" if tau else ("-" if s == 1 else "--")
        mk = "D" if tau else {0.0: "o", 0.5: "s", 1.0: "^"}[frr]
        lab = rf"$f_\mathrm{{rr}}$ = {frr:g}, $s$ = {s:g}" + (rf", $\tau_\mathrm{{rr}}$ = {tau * 1e6:g} µs" if tau else "")
        kw = dict(color=col, ls=ls, lw=1.3 if tau else 1.0, marker=mk, ms=2.4 if tau else 3.4,
                  mfc=col if s == 1 else "white", label=lab, zorder=3 if tau else 2)  # the delayed return lies on f_rr = 0, s = 1
        a1.plot([r["delta_um"] for r in rows], [r["vaporized_kg_m2"] * 1e4 for r in rows], **kw)
        a2.plot([r["delta_um"] for r in rows], [r["peak_substrate_K"] for r in rows], **kw)
        thick = float(np.median([r["vaporized_kg_m2"] for r in rows_all if r["delta_um"] >= 100 and r["status"] == "ok"]))
        r31 = next(r for r in rows_all if abs(r["delta_um"] - 31.6228) < 1e-3)
        T31[tag] = r31["peak_substrate_K"]
        q = TABLE6[tag]
        check(F, f"{tag}: thick-film evaporation (kg m-2)", thick, q[0])
        check(F, f"{tag}: ratio to no shielding", thick / thick0, q[1])
        check(F, f"{tag}: peak steel, 31.6 µm (K)", r31["peak_substrate_K"], q[2])
        check_min(F, f"{tag}: minimum film (µm)", crossing_delta(rows_all), q[3])
    check(F, "no shielding: thick-film evaporation (kg m-2)", thick0, "7.43e-4")
    check(F, "no shielding: peak steel, 31.6 µm (K)", T31["none"], "907")
    check_min(F, "no shielding: minimum film (µm)", crossing_delta(base), "30.1")
    # S2 quotes the lowering as the difference of the peaks printed in Table S4 (whole kelvin, 27 Oct 2026)
    dT = [round(T31["none"]) - round(v) for k, v in T31.items() if k != "none"]
    check(F, "lowering of the peak steel at 31.6 µm, smallest, from the printed peaks (K)", min(dT), "6")
    check(F, "same, largest (K)", max(dT), "19")
    for ax, L in zip((a1, a2), "ab"):
        ax.set_xscale("log")
        ax.set_xlim(5, 1250)  # from the thinnest film every case completes (5.6 µm)
        plain_log(ax.xaxis)
        letter(ax, L)
    a1.set_ylabel(r"Evaporated mass per shot (10$^{-4}$ kg m$^{-2}$)")
    a1.yaxis.set_minor_locator(MultipleLocator(0.25))
    a2.set_ylabel("Peak steel temperature (K)")
    a2.set_ylim(600, 1850)
    a2.yaxis.set_minor_locator(MultipleLocator(50))
    a2.set_xlabel(r"Lithium film thickness $\delta$ (µm)")
    ref_line(a2, T_LIMIT)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=2)
    label_line(F, [a2], T_LIMIT, ["Steel limit 925 K"], xs=(0.98, 0.5, 0.7))
    return fig


# ------------------------------------------------------------------------------------------ Fig. C.1
def fig_c1():
    F = "figC1"
    if not VERIF_CURVES.exists():
        raise MissingData("no stored solver curves: scripts/plot_verification.py runs the solver on the fly and "
                          "runs/verification/results.json holds only error summaries (see the docstring)")
    note_src(F, VERIF_CURVES)
    z = np.load(VERIF_CURVES)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 62 * MM), layout="constrained")
    a.plot(z["a_t_s"], z["a_dT_K"], ls="none", marker="o", mfc="none", color=C["blue"], ms=3, label="Solver")
    a.plot(z["a_t_exact_s"], z["a_dT_exact_K"], color="k", lw=0.9, label="Analytical")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Time (s)")
    a.set_ylabel(r"Surface temperature rise $T_\mathrm{surf}-T_\mathrm{init}$ (K)")  # Table S1 symbols (27 Oct 2026)
    a.legend(loc="upper left")
    cols = (C["blue"], C["orange"], C["green"], C["vermillion"])
    for i, tk in enumerate(z["b_times_s"]):
        x = z[f"b_x_um_{i}"]
        b.plot(x, z[f"b_dT_K_{i}"], ls="none", marker="o", mfc="none", ms=2.6, color=cols[i])
        b.plot(x, z[f"b_dT_exact_K_{i}"], color=cols[i], lw=0.9, label=f"$t$ = {tk * 1e9:g} ns")
    b.set_xscale("log")
    b.set_xlim(1e-3, 30)
    plain_log(b.xaxis)
    b.set_xlabel("Depth (µm)")
    b.set_ylabel(r"Temperature rise $T-T_\mathrm{init}$ (K)")
    b.legend(loc="upper right", handlelength=1.5, borderaxespad=0.3)  # narrower: clear of the 0.1 ns markers
    c.plot(z["c_t_s"], z["c_X_mm"], ls="none", marker="o", mfc="none", ms=3, color=C["blue"], label="Solver")
    c.plot(z["c_t_exact_s"], z["c_X_exact_mm"], color="k", lw=0.9, label="Neumann solution")
    c.set_xlabel("Time (s)")
    c.set_ylabel("Melt depth (mm)")
    c.legend(loc="lower right")
    for ax, L in zip((a, b, c), "abc"):
        letter(ax, L)
    return fig


# ------------------------------------------------------------------------------------------ Fig. D.1
ION_TXT = {"p": "p", "d": "d", "t": "t", "he3": r"$^{3}$He", "he4": r"$^{4}$He", "c12": r"$^{12}$C", "au": "Au"}


def fig_d1():
    F = "figD1"
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from ife_wall.ions import csda_fractions_layered
    from ife_wall.srim import DATA, EV, SPECIES_CODE, SrimStopping, trim_profile
    code2sp = {v: k for k, v in SPECIES_CODE.items()}
    summ = []
    for f in (DATA / "trim" / "summary.csv", DATA / "trim_supplementary" / "summary.csv"):
        note_src(F, f)
        with open(f, newline="", encoding="utf-8") as fh:
            summ += list(csv.DictReader(fh))
    note_src(F, DATA / "trim" / "<run>" / "IONIZ.txt, PHONON.txt")
    note_src(F, DATA / "<ion>_in_<target>.txt (SRIM tables)")
    stored = {r["run"]: r for r in jload(F, "runs/validation/srim_checks.json")["trim_runs"]}
    stop = {m: SrimStopping(mode=m) for m in ("projected", "csda")}
    nrow, ncol = 3, 5
    fig, axs = plt.subplots(nrow, ncol, figsize=(W2, 150 * MM), layout="constrained")
    ratio, labels = {}, []
    for k, (ax, r) in enumerate(zip(axs.flat, summ)):
        run_dir = DATA / "trim" / r["run"] if (DATA / "trim" / r["run"]).exists() else DATA / "trim_supplementary" / r["run"]
        x, q, w = trim_profile(run_dir)
        E0 = float(r["E_keV"]) * 1e3 * EV
        c = np.cumsum(q * w)
        x50 = float(np.interp(0.5 * c[-1], c, x + w / 2))
        x90 = float(np.interp(0.9 * c[-1], c, x + w / 2))
        rho, thick, el = float(r["density_g_cm3"]) * 1e3, float(r["thickness_um"]) * 1e-6, r["target"]
        xmax = min(thick, 2.2 * x90) * 1e6
        ytop = [np.max((q / E0 * 1e-6)[x * 1e6 <= xmax])]
        ax.plot(x * 1e6, q / E0 * 1e-6, color="k", lw=1.0, label="TRIM", zorder=4)
        nc = 2000
        m = np.full(nc, thick * rho / nc)
        xc = (np.arange(nc) + 0.5) * thick / nc
        wc = thick / nc
        for mode, col, ls, lab in (("projected", C["blue"], "-", "Projected depth model (production runs)"),
                                   ("csda", C["vermillion"], "--", "CSDA depth model")):
            fr = csda_fractions_layered(E0, code2sp[r["ion"]], m, np.zeros(nc, int), [el], stop[mode])
            cc = np.cumsum(fr)
            ratio[(r["run"], mode)] = float(np.interp(0.5 * cc[-1], cc, xc + wc / 2)) / x50
            prof = fr / wc / 1e6
            ytop.append(np.max(prof[xc * 1e6 <= xmax]))
            ax.plot(xc * 1e6, prof, color=col, ls=ls, lw=0.8, label=lab, zorder=3 if mode == "projected" else 2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.42 * max(ytop))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(NoZeroFormatter())
        ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)
        ident = r["run"].split("_")[0] + (", full cascade" if "fullcascade" in r["run"] else "")
        E = float(r["E_keV"])
        etxt = f"{E:g} keV" if E < 1000 else f"{E / 1000:g} MeV"
        labels.append((ax, f"({chr(97 + k)}) {ident}\n{ION_TXT.get(r['ion'], r['ion'])}, {etxt}, {el}"))
    for ax in axs.flat[len(summ):]:
        ax.set_visible(False)
    fig.supxlabel("Depth (µm)", fontsize=FS_LAB)
    fig.supylabel(r"Energy deposited per unit depth / $E_0$ (µm$^{-1}$)", fontsize=FS_LAB)
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncols=3)
    fig.canvas.draw()
    px = fig.dpi / 72.0
    for ax, s in labels:  # panel letter and case, top right if free; inset 1.5 pt beyond the inward ticks
        bb = ax.get_window_extent()
        ix = (plt.rcParams["ytick.major.size"] + 1.5) * px / bb.width
        iy = (plt.rcParams["xtick.major.size"] + 1.5) * px / bb.height
        opts = [(s, 1 - ix, 1 - iy, "right", "top", "axes", 0), (s, 0.5, 1 - iy, "center", "top", "axes", 0),
                (s, ix, 1 - iy, "left", "top", "axes", 0)]
        place(ax, opts, fallback="first", draw=False, multialignment="right")
    dev = max(abs(ratio[(n, "projected")] - stored[n]["projected_x50_over_trim"]) for n in stored)
    check_true(F, "recomputed 50 %-depth ratios equal runs/validation/srim_checks.json", dev < 1e-9, f"max diff {dev:.1e}")
    low = [ratio[(n, "projected")] for n in ("B1", "B2", "B3", "B4", "S1_he4_1keV_W", "S2_he4_10keV_W")]
    check(F, "projected/TRIM 50 %-depth, d and He 1-100 keV in W, low end", min(low), "0.69")
    check(F, "same, high end", max(low), "0.76")
    hi_ = [ratio[(f"B{i}", "projected")] for i in range(5, 13)]
    check(F, "projected/TRIM 50 %-depth, B5-B12 (Table E.1), low end", min(hi_), "0.935")
    check(F, "same, high end", max(hi_), "1.005")
    return fig


# ------------------------------------------------------------------------------------------ Fig. E.1
def fig_e1():
    """Helium reaching the steel per shot against the film thickness (revision of 21 Oct 2026):
    the fine grid of scripts/steel_behind_film.py, with the 10^6-shot levels and the 925 K minimum films."""
    F = "figE1"
    sb = jload(F, "results/revision/steel_behind_film/steel_behind_film.json")
    crit = jload(F, "results/revision/steel_behind_film/steel_criteria.json")
    # the 925 K minima of Table 5, read from the saved outputs (repeated shots; Experimental single shot)
    minima = {"Experimental": crossing_delta(jload(F, "runs/li_scan/T600K/Experimental.json")["rows"]),
              "Prototype": jload(F, "results/revision/li_compatibility/li_compatibility.json")[
                  "Prototype"]["repeated_shot_min_film_um"]["925 K"],
              "Demo": jload(F, "results/revision/li_direct/minima_direct.json")[
                  "minima"]["production_and_direct"]["925K"]["delta_um"]}
    for s, q in (("Experimental", "7.2"), ("Prototype", "12.1"), ("Demo", "44")):
        check_min(F, f"{s}: 925 K minimum film marked (µm)", minima[s], q)
    fig, ax = plt.subplots(figsize=(W1, 66 * MM), layout="constrained")
    for s in SCEN:
        g = sb["scenarios"][s]["fine_grid"]
        d, he = np.array(g["delta_um"]), np.array(g["He_per_m2_per_shot"])
        ok = (he > 0) & (d <= 1000)
        col, ls, mk = STY_SCEN[s]
        ax.plot(d[ok], he[ok], color=col, ls=ls, lw=1.3, label=s)
        ax.plot([minima[s]], [np.interp(np.log(minima[s]), np.log(d), he)], color=col, marker=mk, ms=4.5, ls="none")
    for lv, txt in ((1e12, r"10$^6$ shots to 10$^{18}$ m$^{-2}$"), (1e13, r"10$^6$ shots to 10$^{19}$ m$^{-2}$")):
        ax.axhline(lv, color=C["gray"], lw=0.8, ls=":")
        ax.text(1.15, lv * 1.9, txt, color=C["gray"], fontsize=FS)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e9, 3e17)
    plain_log(ax.xaxis)
    ax.set_xlabel(r"Lithium film thickness $\delta$ (µm)")
    ax.set_ylabel(r"Helium reaching the steel (m$^{-2}$ per shot)")
    ax.legend(loc="upper right")
    check_min(F, "Demo film for 1e6 shots to 1e18 m-2 (um)", crit["scenarios"]["Demo"]["film_um"]["1e6_shots_to_1e18"], "184")
    check_min(F, "Prototype film for 1e6 shots to 1e18 m-2 (um)", crit["scenarios"]["Prototype"]["film_um"]["1e6_shots_to_1e18"], "174")
    check_min(F, "Experimental film for 1e6 shots to 1e18 m-2 (um)", crit["scenarios"]["Experimental"]["film_um"]["1e6_shots_to_1e18"], "165")
    check_min(F, "film that stops every helium ion (mm)", crit["scenarios"]["Demo"]["film_stopping_all_He_um"] / 1e3, "0.69")
    return fig


# ------------------------------------------------------------------------------------------ Fig. H.1
LI_CASES = [("T600K_alpha_evap-0.1", "evaporation coefficient 0.1"), ("T500K", "film and coolant at 500 K"),
            ("T800K", "film and coolant at 800 K"), ("T600K_xray_tau-1e-10", "x-ray pulse 0.1 ns"),
            ("T600K_xray_tau-5e-08", "x-ray pulse 50 ns")]
# left out: runs/li_scan/T600K_lv-handbook_147kJmol (alternative Li heat of
# vaporization, source not found).  T600K_dx0-1e-9 and T600K_steel_thickness-0.0035 postdate the figure.


def fig_h1():
    F = "figH1"

    def thick(j):
        return float(np.median([r["vaporized_kg_m2"] for r in j["rows"] if r["delta_um"] >= 100]))
    b = jload(F, "runs/li_scan/T600K/Demo.json")
    v0, d0 = thick(b), crossing_delta(b["rows"])
    labs, lv, ld, rv, rd = [], [], [], [], []
    for tag, lab in LI_CASES:
        j = jload(F, f"runs/li_scan/{tag}/Demo.json")
        labs.append(lab)
        rv.append(thick(j) / v0)
        rd.append(crossing_delta(j["rows"]) / d0)
        lv.append(np.log10(rv[-1]))
        ld.append(np.log10(rd[-1]))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 90 * MM), layout="constrained")
    tornado(a1, labs, lv, lambda i: _ratio_txt(rv[i]))
    a1.set_xlabel("log$_{10}$ of the thick-film evaporation ratio\nto the baseline of " + sci_tex(v0) + " kg m$^{-2}$")
    letter(a1, "a")
    tornado(a2, labs, ld, lambda i: _ratio_txt(rd[i]))
    a2.set_xlabel(f"log$_{{10}}$ of the minimum-film ratio\nto the baseline of {d0:.1f} µm")
    letter(a2, "b")
    check(F, "baseline thick-film evaporation (kg m-2)", v0, "7.43e-4")
    check_min(F, "baseline minimum film (µm)", d0, "30.1")
    mins = dict(zip([t for t, _ in LI_CASES], [x * d0 for x in rd]))
    check_min(F, "minimum film, evaporation coefficient 0.1 (µm)", mins["T600K_alpha_evap-0.1"], "36")
    allv = [v0] + [x * v0 for x in rv]
    alld = [d0] + list(mins.values())
    check_min(F, "Table 5 range of the minimum film, low end (µm)", min(alld), "24")
    check_min(F, "Table 5 range of the minimum film, high end (µm)", max(alld), "79")
    check(F, "Table 5 range of the evaporated mass, low end (kg m-2)", min(allv), "4.2e-4")
    check(F, "Table 5 range of the evaporated mass, high end (kg m-2)", max(allv), "9.5e-4")
    return fig


# ------------------------------------------------------------------------------------------ Fig. H.2
GRADES = [  # variant, label, color, line style, line width, z-order (coinciding curves stay visible)
    ("W_ODS_La2O3_p1.0", r"ODS W–La$_2$O$_3$, $\varphi$ = 1.0", C["vermillion"], "-", 2.4, 2),
    ("W_pure", "pure W", C["black"], "-", 1.0, 3),
    ("W_K", "W–K", C["blue"], (0, (4, 3)), 1.1, 4),
    ("W_ODS_La2O3_p0.9", r"ODS W–La$_2$O$_3$, $\varphi$ = 0.9", C["vermillion"], "--", 1.9, 2),
    ("W_K_k0.9", r"W–K, $\varphi$ = 0.9", C["blue"], ":", 1.3, 4),
    ("W_ODS_La2O3_p0.8", r"ODS W–La$_2$O$_3$, $\varphi$ = 0.8", C["vermillion"], (0, (1, 1.2)), 1.3, 3),
    ("W_K_k1.1", r"W–K, $\varphi$ = 1.1", C["blue"], "-.", 1.1, 3)]
LEGEND_ORDER = ["W_pure", "W_K", "W_K_k0.9", "W_K_k1.1", "W_ODS_La2O3_p1.0", "W_ODS_La2O3_p0.9", "W_ODS_La2O3_p0.8"]
ONSET = [("W_pure", "pure W", C["black"]), ("W_K", "W–K", C["blue"]), ("W_ODS_La2O3_p1.0", r"ODS W–La$_2$O$_3$", C["vermillion"])]


def fig_h2():
    F = "figH2"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 132 * MM), layout="constrained")
    js, handles = {}, {}
    for v, lab, col, ls, lw, z in GRADES:
        j = js[v] = jload(F, f"results/alloys/single/{v}/Demo_hot.json")
        h = cload(F, f"results/alloys/single/{v}/Demo_hot_history.csv")
        p = cload(F, f"results/alloys/single/{v}/Demo_hot_maxT_profile.csv")
        handles[v] = a1.plot(h[:, 0] * 1e6, h[:, 1], color=col, ls=ls, lw=lw, zorder=z, label=lab)[0]
        a2.plot(p[:, 0], p[:, 1], color=col, ls=ls, lw=lw, zorder=z, label=lab)
        check(F, f"{v}: peak from the history equals the JSON (K)", float(h[:, 1].max()), f"{j['peak_surface_T_K']:.3f}")
    a1.set_xscale("log")
    a1.set_xlim(1e-2, 50)
    plain_log(a1.xaxis)
    a1.set_ylim(800, 3950)
    a1.yaxis.set_minor_locator(MultipleLocator(250))
    a1.set_xlabel("Time after ignition (µs)")
    a1.set_ylabel("Surface temperature (K)")
    letter(a1, "a")
    ref_line(a1, T_MELT)
    a2.set_xscale("log")
    a2.set_xlim(1e-3, 200)
    plain_log(a2.xaxis)
    a2.set_ylim(800, 3950)
    a2.yaxis.set_minor_locator(MultipleLocator(250))
    a2.set_xlabel("Depth (µm)")
    a2.set_ylabel("Maximum temperature (K)")
    letter(a2, "b")
    # recrystallization onsets (Table B.2): thin solid lines in the grade colors.  They lie 100-300 K apart,
    # less than one line of text at this scale, so they are labeled by a key inside the panel, listed in
    # the order of the lines (highest first)
    onsets = sorted([(js[v]["rx_spec"]["T_ref_K"], name, col) for v, name, col in ONSET], reverse=True)
    keys = [a2.axhline(T, color=col, lw=0.8, zorder=1, label=f"{name}, {T:.0f} K") for T, name, col in onsets]
    a2.legend(handles=keys, title="Recrystallization onset", loc="center left", bbox_to_anchor=(0.0, 0.58),
              handlelength=1.8, alignment="left")
    fig.legend([handles[v] for v in LEGEND_ORDER], [handles[v].get_label() for v in LEGEND_ORDER],
               loc="outside lower center", ncols=2)
    label_line(F, [a1], T_MELT, ["Melting point 3695 K"], xs=(0.02, 0.98, 0.3, 0.7))
    for v, q in (("W_pure", "3317"), ("W_K", "3317"), ("W_ODS_La2O3_p1.0", "3317"), ("W_ODS_La2O3_p0.9", "3429"),
                 ("W_ODS_La2O3_p0.8", "3558")):
        check(F, f"{v}: peak surface (K), Table H.2", js[v]["peak_surface_T_K"], q)
    check(F, "start temperature (K)", js["W_pure"]["T_init_K"], "921")
    for v, q in (("W_pure", "1523"), ("W_K", "1623"), ("W_ODS_La2O3_p1.0", "1823")):
        check(F, f"{v}: onset temperature (K), Table B.2", js[v]["rx_spec"]["T_ref_K"], q)
    return fig


# ------------------------------------------------------------------------------------------ Fig. H.3
def fig_h3():
    F = "figH3"
    A = jload(F, "results/alloys/summary.json")
    sb = A["side_by_side"]["Demo"]
    rows = {r["variant"]: r for r in A["single"] if r["scenario"] == "Demo" and r["start"] == "hot"}
    items = [(f"Retained He reaches\n10$^{{18}}$ m$^{{-2}}$", sb["W_He_shots_1e18"], None, C["green"]),
             (f"Retained He reaches\n10$^{{19}}$ m$^{{-2}}$", sb["W_He_shots_1e19"], None, C["green"])]
    for v, name, col in (("W_pure", "pure W", C["black"]), ("W_K", "W–K", C["blue"]),
                         ("W_ODS_La2O3_p1.0", r"ODS W–La$_2$O$_3$, $\varphi$ = 1.0", C["vermillion"]),
                         ("W_ODS_La2O3_p0.8", r"ODS W–La$_2$O$_3$, $\varphi$ = 0.8", C["vermillion"])):
        r = rows[v]
        band = [r[k] for k in ("shots_to_onset_surface_Q322", "shots_to_onset_surface_Q350",
                               "shots_to_onset_surface_Trx_m100", "shots_to_onset_surface_Trx_p100")]
        items.append((f"Recrystallization,\n{name}", r["shots_to_onset_surface"], (min(band), max(band)), col))
    items.append(("1 mm lost to evaporation\nand sputtering (pure W)", sb["W_shots_1mm_evap_plus_sputter_pure"], None, C["gray"]))
    fig, ax = plt.subplots(figsize=(W1, 86 * MM), layout="constrained")
    y = np.arange(len(items))[::-1]
    for yy, (lab, v, rng, col) in zip(y, items):
        ax.barh(yy, v - 1.0, left=1.0, height=0.62, color=col, alpha=0.85, lw=0)
        if rng:
            ax.errorbar(v, yy, xerr=[[v - rng[0]], [rng[1] - v]], fmt="none", ecolor="k", elinewidth=0.8, capsize=2.2)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e11)
    ax.xaxis.set_major_locator(LogLocator(base=100, numticks=7))
    ax.xaxis.set_minor_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, [it[0] for it in items])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel(r"Number of shots (at 10 Hz, 10$^{4}$ shots = 17 min)")
    check(F, "He reaches 1e18 m-2 (shots)", sb["W_He_shots_1e18"], "10")
    check(F, "He reaches 1e19 m-2 (shots)", sb["W_He_shots_1e19"], "98")
    for v, q in (("W_pure", ("2.2e3", "1.2e3", "3.9e3", "3.4e2", "1.1e4")), ("W_K", ("1.1e4", "6.8e3", "1.9e4", "2.2e3", "4.8e4")),
                 ("W_ODS_La2O3_p1.0", ("1.7e5", "1.2e5", "2.6e5", "4.8e4", "5.5e5")),
                 ("W_ODS_La2O3_p0.8", ("7.4e4", "4.9e4", "1.1e5", "2.1e4", "2.4e5"))):
        r = rows[v]
        qs = sorted([r["shots_to_onset_surface_Q322"], r["shots_to_onset_surface_Q350"]])
        for what, val, qq in (("shots to the one-hour exposure", r["shots_to_onset_surface"], q[0]),
                              ("Q = 350 kJ/mol", qs[0], q[1]), ("Q = 322 kJ/mol", qs[1], q[2]),
                              ("onset -100 K", r["shots_to_onset_surface_Trx_m100"], q[3]),
                              ("onset +100 K", r["shots_to_onset_surface_Trx_p100"], q[4])):
            check(F, f"{v}: {what}, Table H.2", val, qq)
    fpy = sb["W_shots_1mm_evap_plus_sputter_pure"] / (10 * 3.156e7)
    check(F, "1 mm by evaporation + sputtering, full-power years (Sec. 4.4)", fpy, "27")
    return fig


# ------------------------------------------------------------------------------------------ checks of the result
def _texts(fig):
    out, seen = [], set()

    def add(t, kind, ax=None):
        if t is not None and id(t) not in seen and t.get_visible() and t.get_text().strip():
            seen.add(id(t))
            out.append((t, kind, ax))
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis._update_ticks():
                for lab in (tk.label1, tk.label2):
                    add(lab, "tick", ax)
            add(axis.label, "axislabel", ax)
            add(axis.get_offset_text(), "offset", ax)
        for t in (ax.title, ax._left_title, ax._right_title):
            add(t, "title", ax)
        for t in ax.texts:
            add(t, "annot", ax)
        if ax.get_legend() is not None:
            for t in [*ax.get_legend().get_texts(), ax.get_legend().get_title()]:
                add(t, "legend", ax)
    for lg in fig.legends:
        for t in [*lg.get_texts(), lg.get_title()]:
            add(t, "figlegend")
    for t in [*fig.texts, getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None), getattr(fig, "_suptitle", None)]:
        add(t, "figtext")
    return out


def layout_problems(fig):
    """Font sizes, texts outside the figure, overlaps between texts, legends and data."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb, probs, items = fig.bbox, [], []
    for t, kind, ax in _texts(fig):
        s = t.get_text()
        if t.get_fontsize() < FS - 1e-6:
            probs.append(f"{t.get_fontsize():g} pt: {s!r}")
        if "vapour" in s.lower():
            probs.append(f"British spelling: {s!r}")
        bb = t.get_window_extent(r)
        if not _inside(bb, fb):
            probs.append(f"outside the figure: {s!r}")
        if kind == "annot" and not _inside(bb, ax.get_window_extent(r)):
            probs.append(f"label leaves its axes: {s!r}")
        items.append((t, kind, ax, bb))
    for i in range(len(items)):
        for k in range(i + 1, len(items)):
            if _overlap(items[i][3], items[k][3]):
                probs.append(f"texts overlap: {items[i][0].get_text()!r} / {items[k][0].get_text()!r}")
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        boxes = [(t.get_text(), bb) for t, kind, a2, bb in items if a2 is ax and kind == "annot"]
        if ax.get_legend() is not None:
            lb = ax.get_legend().get_window_extent(r)
            boxes.append(("legend", lb))
            if not _inside(lb, ax.get_window_extent(r)):
                probs.append("legend leaves its axes")
            for t, kind, a2, bb in items:
                if a2 is ax and kind == "annot" and _overlap(lb, bb):
                    probs.append(f"legend overlaps {t.get_text()!r}")
        for name, bb in boxes:
            hit = _hits_data(ax, bb, r)
            if hit:
                probs.append(f"{name!r} touches {hit}")
    for lg in fig.legends:
        lb = lg.get_window_extent(r)
        if not _inside(lb, fb):
            probs.append("figure legend outside the figure")
        for ax in fig.axes:
            if ax.get_visible() and _overlap(lb, ax.get_window_extent(r)):
                probs.append("figure legend overlaps a panel")
        for t, kind, ax, bb in items:
            if kind != "figlegend" and _overlap(lb, bb):
                probs.append(f"figure legend overlaps {t.get_text()!r}")
    return probs


def save(fig, name, out):
    w_in, h_in = fig.get_size_inches()
    pdf, png = out / f"{name}.pdf", out / f"{name}.png"
    fig.savefig(pdf, metadata={"Creator": "scripts/make_article_figures.py", "CreationDate": None})
    fig.savefig(png, dpi=600, metadata={"Software": "scripts/make_article_figures.py (matplotlib)"})
    data = pdf.read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data)
    wp, hp = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    pw, ph = struct.unpack(">II", png.read_bytes()[16:24])
    probs = []
    if abs(wp - w_in * 72) > 0.5 or abs(hp - h_in * 72) > 0.5:
        probs.append(f"PDF page {wp / 72 * 25.4:.1f} x {hp / 72 * 25.4:.1f} mm")
    if re.search(rb"/Subtype\s*/Type3", data) or not re.search(rb"/FontFile2", data):
        probs.append("PDF fonts are not all embedded TrueType (Type 42)")
    if abs(pw - round(w_in * 600)) > 1 or abs(ph - round(h_in * 600)) > 1:
        probs.append(f"PNG {pw} x {ph} px")
    return f"{w_in * 25.4:.0f} x {h_in * 25.4:.0f} mm, PNG {pw} x {ph} px", probs


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7,
           "fig8": fig8, "figC1": fig_c1, "figD1": fig_d1, "figE1": fig_e1, "figH1": fig_h1, "figH2": fig_h2, "figH3": fig_h3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default: docs/figures/article)")
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. fig1,figD1")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    t_all, failed, skipped = time.time(), [], []
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            fig = fn()
        except MissingData as e:
            skipped.append(name)
            print(f"{name}: SKIPPED - {e}")
            continue
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"{name}: FAILED - {type(e).__name__}: {e}")
            continue
        fig.canvas.draw()                # final constrained layout ...
        fig.set_layout_engine("none")    # ... kept for the check and both files (positions do not depend on dpi)
        probs = WARN.get(name, []) + layout_problems(fig)
        info, p2 = save(fig, name, out)
        plt.close(fig)
        probs += p2
        print(f"{name}: {info}, {time.time() - t0:.1f} s; layout {'OK' if not probs else 'PROBLEMS:'}")
        for p in probs:
            print(f"    - {p}")
        for s in SOURCES.get(name, []):
            print(f"    < {s}")
    bad = [c for c in CHECKS if not c[4]]
    print(f"\nPlotted numbers against the article: {len(CHECKS) - len(bad)} of {len(CHECKS)} agree.")
    for fid, what, val, quoted, ok in CHECKS:
        print(f"  {'ok ' if ok else 'BAD'} {fid:6s} {what}: plotted {val}" + (f", text {quoted}" if quoted else ""))
    print(f"\n{len(FIGURES) if not only else len(only)} figure(s) requested, skipped {skipped or 'none'}, "
          f"failed {failed or 'none'}; {time.time() - t_all:.1f} s; output in {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
