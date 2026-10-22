"""Closed-form solutions used by tests/test_analytical.py (phase 4 of the pre-submission verification,
17 Oct 2026).  Every function returns the temperature RISE T - T0 (or the quantity named) in SI units.

Sources
-------
[CJ]  H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids, 2nd ed., Oxford 1959.
      Chap. II  linear flow in the semi-infinite solid: the section "flux at the surface a prescribed function of
                time" (Sec. 2.9 in the 2nd-edition numbering as recalled; the chapter is certain), constant-flux case;
      Chap. X   use of sources and sinks in cases of variable temperature: the instantaneous plane source in an
                infinite solid (Secs. 10.2-10.3);
      Chap. XI  change of state: the one-phase problem (medium initially at the fusion temperature) and Neumann's
                two-phase solution (Secs. 11.2-11.3).
[IdW] F. P. Incropera and D. P. DeWitt, Fundamentals of Heat and Mass Transfer, Chap. 5 (transient conduction),
      "semi-infinite solid", Case 2 (constant surface heat flux) - the same formula as [CJ], a second reference.
[AS]  V. Alexiades and A. D. Solomon, Mathematical Modeling of Melting and Freezing Processes, Hemisphere 1993,
      Chap. 2 (problems with explicit solutions; Sec. 2.1 one-phase Stefan problem, the Neumann similarity
      solution with the transcendental equation lam exp(lam^2) erf(lam) = St / sqrt(pi)).
Because section numbers are quoted from memory, every formula is also derived here in one or two lines from the
plane-source (Gaussian) kernel or the similarity ansatz, so that nothing rests on a section number: see the
docstrings.  Where a formula is a short consequence (the finite-thickness slab deposit, the surface-heat
integral of the Stefan problem) the derivation is likewise given in the docstring.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc


# ----------------------------------------------------------------------------------------------
# 1. Semi-infinite solid, constant surface flux q from t = 0, initially uniform ([CJ] 2.9; [IdW] Ch. 5)
# ----------------------------------------------------------------------------------------------
def constant_flux_profile(x, t, q, k, alpha):
    """T(x,t) - T0 = (2 q / k) sqrt(alpha t / pi) exp(-x^2 / 4 alpha t) - (q x / k) erfc(x / 2 sqrt(alpha t)).
    Surface value: 2 q sqrt(alpha t / pi) / k.
    Derivation: a flux q at the insulated surface of the half-space is, by the image argument, a plane source of
    strength 2 q dt' per unit area in the infinite solid, so T - T0 = int_0^t (q / rho c) exp(-x^2 / 4 alpha (t - t'))
    / sqrt(pi alpha (t - t')) dt'; the integral (Gradshteyn-type, or integrate by parts) is the formula above, and at
    x = 0 it is (2 q / rho c) sqrt(t / pi alpha) = 2 q sqrt(alpha t / pi) / k."""
    x = np.asarray(x, float)
    s = np.sqrt(alpha * t)
    return 2 * q / k * np.sqrt(alpha * t / np.pi) * np.exp(-x ** 2 / (4 * s ** 2)) - q * x / k * erfc(x / (2 * s))


def constant_flux_surface(t, q, k, alpha):
    return 2 * q / k * np.sqrt(alpha * t / np.pi)


# ----------------------------------------------------------------------------------------------
# 2. Instantaneous plane source Q [J/m^2] at the surface of an insulated semi-infinite solid
# ----------------------------------------------------------------------------------------------
def plane_source_profile(x, t, Q, rho_c, alpha):
    """T - T0 = Q / (rho c sqrt(pi alpha t)) exp(-x^2 / 4 alpha t).
    [CJ] Chap. X gives the plane source of strength Q in an INFINITE solid as Q / (2 rho c sqrt(pi alpha t))
    exp(-x^2/4 alpha t) (the Gaussian kernel: it satisfies the heat equation and integrates to Q / rho c);
    the insulated surface at x = 0 is the image plane, which doubles it."""
    x = np.asarray(x, float)
    return Q / (rho_c * np.sqrt(np.pi * alpha * t)) * np.exp(-x ** 2 / (4 * alpha * t))


def slab_source_profile(x, t, Q, d, rho_c, alpha):
    """Same as plane_source_profile but with Q spread uniformly over 0 <= x <= d (the way the code deposits
    a 'surface' energy: UniformDepthSource).  Derivation: integrate the infinite-solid plane-source kernel
    Q/d dx' / (2 rho c sqrt(pi alpha t)) exp(-(x-x')^2 / 4 alpha t) over the slab AND its image -d <= x' <= 0:
        T - T0 = Q / (2 rho c d) [ erf((d - x) / 2 sqrt(alpha t)) + erf((d + x) / 2 sqrt(alpha t)) ].
    At x = 0 this is (Q / rho c d) erf(d / 2 sqrt(alpha t)) -> Q / (rho c sqrt(pi alpha t)) as d -> 0,
    with relative difference -d^2 / (12 alpha t) + O(d^4) (series of erf), i.e. the finite-thickness bias."""
    x = np.asarray(x, float)
    s = np.sqrt(alpha * t)
    return Q / (2 * rho_c * d) * (erf((d - x) / (2 * s)) + erf((d + x) / (2 * s)))


def square_pulse_surface_exact(t, tau, Q, rho_c, alpha):
    """Surface rise for the plane source released uniformly over 0 <= t' <= tau (t >= tau):
    (Q/tau) int_0^tau dt' / (rho c sqrt(pi alpha (t - t'))) = 2 Q [sqrt(t) - sqrt(t - tau)] / (tau rho c sqrt(pi alpha)).
    Used to show that evaluating the instantaneous formula at t - tau/2 leaves an O((tau/t)^2) residual."""
    return 2 * Q * (np.sqrt(t) - np.sqrt(t - tau)) / (tau * rho_c * np.sqrt(np.pi * alpha))


# ----------------------------------------------------------------------------------------------
# 3. One-phase Stefan (Neumann) melting problem ([CJ] 11.2; [AS] 2.1)
# ----------------------------------------------------------------------------------------------
def neumann_one_phase_lambda(St):
    """Root of lam exp(lam^2) erf(lam) = St / sqrt(pi), St = c_l (T_s - T_m) / L  ([AS] Sec. 2.1).
    Derivation: liquid 0 < x < s(t) with T(0,t) = T_s, T(s,t) = T_m, solid at T_m (no flux in the solid);
    similarity solution T = T_s - (T_s - T_m) erf(x / 2 sqrt(alpha t)) / erf(lam), s(t) = 2 lam sqrt(alpha t);
    the Stefan condition rho L ds/dt = -k dT/dx|_s gives the transcendental equation."""
    return brentq(lambda l: l * np.exp(l ** 2) * erf(l) - St / np.sqrt(np.pi), 1e-9, 6.0)


def neumann_two_phase_lambda(St_l, St_s, nu=1.0):
    """Two-phase Neumann root ([CJ] Chap. XI, the form used by tests/test_verification.py test 3):
    St_l exp(-lam^2)/erf(lam) - (St_s/nu) exp(-(nu lam)^2)/erfc(nu lam) = lam sqrt(pi), nu = sqrt(alpha_l/alpha_s).
    Used only to bound the effect of initialising the solid at the solidus T_m - dTm/2 (St_s = c dTm/2 / L)."""
    f = lambda l: St_l * np.exp(-l ** 2) / erf(l) - St_s / nu * np.exp(-(nu * l) ** 2) / erfc(nu * l) - l * np.sqrt(np.pi)
    return brentq(f, 1e-9, 6.0)


def neumann_front(t, lam, alpha):
    return 2 * lam * np.sqrt(alpha * t)


def neumann_liquid_profile(x, t, T_s, T_m, lam, alpha):
    """T(x,t) in the liquid 0 <= x <= s(t)."""
    return T_s - (T_s - T_m) * erf(np.asarray(x, float) / (2 * np.sqrt(alpha * t))) / erf(lam)


def neumann_heat_input(t, k, T_s, T_m, lam, alpha):
    """Cumulative heat through the surface, int_0^t k (-dT/dx)|_0 dt' = 2 k (T_s - T_m) sqrt(t) / (erf(lam) sqrt(pi alpha)).
    Derivation: -dT/dx|_0 = (T_s - T_m) / (erf(lam) sqrt(pi alpha t)); integrate in t."""
    return 2 * k * (T_s - T_m) * np.sqrt(t) / (erf(lam) * np.sqrt(np.pi * alpha))


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def observed_orders(errors):
    """log2 ratios of successive |errors| for a sequence obtained by halving the discretisation parameter."""
    e = np.abs(np.asarray(errors, float))
    return [float(np.log2(e[i] / e[i + 1])) if e[i + 1] > 0 else float("inf") for i in range(len(e) - 1)]


def nested_geometric_mesh(level, dx0, r0):
    """Mesh family in which every cell of level l is split into two at level l+1:
    r -> sqrt(r), dx_first -> dx_first / (1 + sqrt(r)).  (The two children of a cell of width w are
    w / (1 + sqrt r) and w sqrt r / (1 + sqrt r), so the new first cell and ratio follow.)"""
    r, dx = r0, dx0
    for _ in range(level):
        dx = dx / (1 + np.sqrt(r))
        r = np.sqrt(r)
    return dx, r
