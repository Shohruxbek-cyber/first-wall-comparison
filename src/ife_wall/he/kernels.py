"""Numba kernels: right-hand side and banded Jacobian of the He/vacancy/SIA reaction-diffusion system.

State layout: y[0:3] = cumulative surface losses of He, V, I; y[3:6] = cumulative He implanted,
vacancies produced, interstitials produced (all appm um); then, for every cell i and species s,
y[6 + i*S + s].  Species: 0 He, 1 V, 2 I, 3..2+K = HV_1..HV_K, then (Level 2)
3+K..2+K+J = He_2..He_{J+1} clusters.  Concentrations in appm, lengths in um, time in s.

The equations are written in ife_wall/he/model.py (docstring); this file is the compiled form.
`rhs_np` in model.py is an independent NumPy implementation used only for cross-checks.
"""
from __future__ import annotations

import numpy as np
from numba import njit

KB = 8.617333262e-5  # eV/K
H = 6  # header: surface losses He,V,I (0-2), cumulative He implanted, V produced, I produced (3-5)

# parameter vector layout
(P_D0He, P_EmHe, P_D0V, P_EmV, P_D0I, P_EmI, P_rc, P_riv, P_nu, P_SHe, P_SV, P_SI, P_surf, P_ann,
 P_NC, P_L2, P_nuHeHe, P_rHe1) = range(18)
NPAR = 18


@njit(cache=True, inline="always")
def _add(ab, mu, r, c, v):
    ab[mu + r - c, c] += v


@njit(cache=True)
def _interp_T(t, tT, Tc, T):
    nt = tT.shape[0]
    if t <= tT[0]:
        k, f = 0, 0.0
    elif t >= tT[nt - 1]:
        k, f = nt - 2, 1.0
    else:
        k = np.searchsorted(tT, t) - 1
        f = (t - tT[k]) / (tT[k + 1] - tT[k])
    for i in range(T.shape[0]):
        T[i] = Tc[k, i] * (1.0 - f) + Tc[k + 1, i] * f


@njit(cache=True)
def _interp_src(t, tN, G, out):
    m = tN.shape[0]
    if t <= tN[0] or t >= tN[m - 1]:
        for i in range(out.shape[0]):
            out[i] = 0.0
        return
    k = np.searchsorted(tN, t) - 1
    f = (t - tN[k]) / (tN[k + 1] - tN[k])
    for i in range(out.shape[0]):
        out[i] = G[k, i] * (1.0 - f) + G[k + 1, i] * f


@njit(cache=True)
def _rates_cell(T, prm, Eb, K, J, EbC, rCl, nutm, Etm, a, b, c_rate, e_rate, out):
    """per-cell rate coefficients.  out = D_He, D_V, D_I (um^2/s)"""
    kT = KB * T
    DHe = prm[P_D0He] * np.exp(-prm[P_EmHe] / kT) * 1e12
    DV = prm[P_D0V] * np.exp(-prm[P_EmV] / kT) * 1e12
    DI = prm[P_D0I] * np.exp(-prm[P_EmI] / kT) * 1e12
    out[0], out[1], out[2] = DHe, DV, DI
    k4 = 4.0 * np.pi * 1e-12 * prm[P_NC]
    a[0] = k4 * prm[P_rc] * (DHe + DV)
    for k in range(1, K):
        a[k] = k4 * prm[P_rc] * DHe
    for k in range(K):
        b[k] = prm[P_nu] * np.exp(-(Eb[k] + prm[P_EmHe]) / kT)
    c_rate[0] = k4 * prm[P_riv] * (DI + DV)
    e_rate[0] = k4 * prm[P_riv] * DI * prm[P_ann]


@njit(cache=True)
def rhs_nb(t, y, prm, Eb, dx, tT, Tc, tN, GHe, GV, bgHe, bgV, K, J, EbC, rCl, nutm, Etm, out):
    n = dx.shape[0]
    S = 3 + K + J
    T = np.empty(n)
    gHe = np.empty(n)
    gV = np.empty(n)
    _interp_T(t, tT, Tc, T)
    _interp_src(t, tN, GHe, gHe)
    _interp_src(t, tN, GV, gV)
    for i in range(n):
        gHe[i] += bgHe[i]
        gV[i] += bgV[i]
    DHe = np.empty(n); DV = np.empty(n); DI = np.empty(n)
    a = np.empty((n, K)); b = np.empty((n, K)); cc = np.empty(n); ee = np.empty(n)
    tmp3 = np.empty(3); ar = np.empty(K); br = np.empty(K); cr = np.empty(1); er = np.empty(1)
    for i in range(n):
        _rates_cell(T[i], prm, Eb, K, J, EbC, rCl, nutm, Etm, ar, br, cr, er, tmp3)
        DHe[i], DV[i], DI[i] = tmp3[0], tmp3[1], tmp3[2]
        for k in range(K):
            a[i, k] = ar[k]; b[i, k] = br[k]
        cc[i] = cr[0]; ee[i] = er[0]
    for j in range(out.shape[0]):
        out[j] = 0.0
    # --- diffusion (conductance form) for species 0,1,2
    for s in range(3):
        for i in range(n - 1):
            if s == 0:
                Di, Dj = DHe[i], DHe[i + 1]
            elif s == 1:
                Di, Dj = DV[i], DV[i + 1]
            else:
                Di, Dj = DI[i], DI[i + 1]
            Gc = 1.0 / (dx[i] / (2.0 * Di) + dx[i + 1] / (2.0 * Dj))
            fl = Gc * (y[H + i * S + s] - y[H + (i + 1) * S + s])
            out[H + i * S + s] -= fl / dx[i]
            out[H + (i + 1) * S + s] += fl / dx[i + 1]
    for s in range(3):
        D0 = DHe[0] if s == 0 else (DV[0] if s == 1 else DI[0])
        Gs = prm[P_surf] * 2.0 * D0 / dx[0]
        out[s] = Gs * y[H + s]                          # ledger (surface loss)
        out[H + s] -= Gs * y[H + s] / dx[0]
    # --- reactions, sinks, sources
    K1 = 3 + K
    for i in range(n):
        o = H + i * S
        He, V, I = y[o], y[o + 1], y[o + 2]
        # He + V_k -> V_{k+1}
        for k in range(K):
            Vk = y[o + 1] if k == 0 else y[o + 2 + k]
            rt = a[i, k] * He * Vk
            out[o] -= rt
            if k == 0:
                out[o + 1] -= rt
            else:
                out[o + 2 + k] -= rt
            out[o + 3 + k] += rt                          # V_{k+1}: species index 3 + k  (HV_{k+1} at 2 + (k+1))
        # detrapping V_k -> V_{k-1} + He   (k = 1..K)
        for k in range(1, K + 1):
            rd = b[i, k - 1] * y[o + 2 + k]
            out[o + 2 + k] -= rd
            if k == 1:
                out[o + 1] += rd
            else:
                out[o + 1 + k] += rd
            out[o] += rd
        # recombination V + I
        rec = cc[i] * V * I
        out[o + 1] -= rec
        out[o + 2] -= rec
        # I + V_k -> k He
        if prm[P_ann] > 0.0:
            for k in range(1, K + 1):
                an = ee[i] * y[o + 2 + k] * I
                out[o + 2 + k] -= an
                out[o + 2] -= an
                out[o] += k * an
        # sinks
        out[o] -= DHe[i] * 1e-12 * prm[P_SHe] * He
        out[o + 1] -= DV[i] * 1e-12 * prm[P_SV] * V
        out[o + 2] -= DI[i] * 1e-12 * prm[P_SI] * I
        # sources
        out[o] += gHe[i]
        out[o + 1] += gV[i]
        out[o + 2] += gV[i]
        out[3] += dx[i] * gHe[i]
        out[4] += dx[i] * gV[i]
        out[5] += dx[i] * gV[i]
        # ---- Level 2: He clusters, trap mutation
        if J > 0:
            kT = KB * T[i]
            k4 = 4.0 * np.pi * 1e-12 * prm[P_NC]
            sC0 = 3 + K
            # He + He -> C_2   (rate: 1/2 k He^2 pairs; consumes 2 He)
            kdim = k4 * prm[P_rHe1] * 2.0 * 2.0 * DHe[i]
            rdim = 0.5 * kdim * He * He
            out[o] -= 2.0 * rdim
            out[o + sC0] += rdim
            # C_2 -> 2 He
            r2 = prm[P_nuHeHe] * np.exp(-(EbC[0] + prm[P_EmHe]) / kT) * y[o + sC0]
            out[o + sC0] -= r2
            out[o] += 2.0 * r2
            for jj in range(J):          # cluster of size jj+2
                Cj = y[o + sC0 + jj]
                # capture of a vacancy: C + V -> HV_{size}
                if jj + 2 <= K:
                    rv = k4 * (prm[P_rc] + 0.0) * DV[i] * V * Cj
                    out[o + sC0 + jj] -= rv
                    out[o + 1] -= rv
                    out[o + 2 + (jj + 2)] += rv
                # growth by He capture C_j + He -> C_{j+1}
                if jj + 1 < J:
                    rg = k4 * rCl[jj] * DHe[i] * He * Cj
                    out[o + sC0 + jj] -= rg
                    out[o] -= rg
                    out[o + sC0 + jj + 1] += rg
                    # detrapping C_{j+1} -> C_j + He
                    rb = prm[P_nuHeHe] * np.exp(-(EbC[jj + 1] + prm[P_EmHe]) / kT) * y[o + sC0 + jj + 1]
                    out[o + sC0 + jj + 1] -= rb
                    out[o + sC0 + jj] += rb
                    out[o] += rb
                # trap mutation C_size -> HV_size + I
                if nutm[jj] > 0.0 and jj + 2 <= K:
                    rm = nutm[jj] * np.exp(-Etm[jj] / kT) * Cj
                    out[o + sC0 + jj] -= rm
                    out[o + 2 + (jj + 2)] += rm
                    out[o + 2] += rm


@njit(cache=True)
def jac_nb(t, y, prm, Eb, dx, tT, Tc, K, J, EbC, rCl, nutm, Etm, mu, ab):
    n = dx.shape[0]
    S = 3 + K + J
    T = np.empty(n)
    _interp_T(t, tT, Tc, T)
    DHe = np.empty(n); DV = np.empty(n); DI = np.empty(n)
    a = np.empty((n, K)); b = np.empty((n, K)); cc = np.empty(n); ee = np.empty(n)
    tmp3 = np.empty(3); ar = np.empty(K); br = np.empty(K); cr = np.empty(1); er = np.empty(1)
    for i in range(n):
        _rates_cell(T[i], prm, Eb, K, J, EbC, rCl, nutm, Etm, ar, br, cr, er, tmp3)
        DHe[i], DV[i], DI[i] = tmp3[0], tmp3[1], tmp3[2]
        for k in range(K):
            a[i, k] = ar[k]; b[i, k] = br[k]
        cc[i] = cr[0]; ee[i] = er[0]
    ab[:, :] = 0.0
    # diffusion
    for s in range(3):
        for i in range(n - 1):
            if s == 0:
                Di, Dj = DHe[i], DHe[i + 1]
            elif s == 1:
                Di, Dj = DV[i], DV[i + 1]
            else:
                Di, Dj = DI[i], DI[i + 1]
            Gc = 1.0 / (dx[i] / (2.0 * Di) + dx[i + 1] / (2.0 * Dj))
            ri, rj = H + i * S + s, H + (i + 1) * S + s
            _add(ab, mu, ri, ri, -Gc / dx[i])
            _add(ab, mu, ri, rj, Gc / dx[i])
            _add(ab, mu, rj, rj, -Gc / dx[i + 1])
            _add(ab, mu, rj, ri, Gc / dx[i + 1])
    for s in range(3):
        D0 = DHe[0] if s == 0 else (DV[0] if s == 1 else DI[0])
        Gs = prm[P_surf] * 2.0 * D0 / dx[0]
        r0 = s
        c0 = H + s
        _add(ab, mu, c0, c0, -Gs / dx[0])
        _add(ab, mu, r0, c0, Gs)                       # ledger row
    for i in range(n):
        o = H + i * S
        He, V, I = y[o], y[o + 1], y[o + 2]
        for k in range(K):
            ck = o + 1 if k == 0 else o + 2 + k                      # column of V_k
            dHe_ = a[i, k] * (y[o + 1] if k == 0 else y[o + 2 + k])
            dV_ = a[i, k] * He
            _add(ab, mu, o, o, -dHe_); _add(ab, mu, o, ck, -dV_)
            _add(ab, mu, ck, o, -dHe_); _add(ab, mu, ck, ck, -dV_)
            rn = o + 3 + k                                           # HV_{k+1}
            _add(ab, mu, rn, o, dHe_); _add(ab, mu, rn, ck, dV_)
        for k in range(1, K + 1):
            d = b[i, k - 1]
            ck = o + 2 + k
            _add(ab, mu, ck, ck, -d)
            pv = o + 1 if k == 1 else o + 1 + k
            _add(ab, mu, pv, ck, d)
            _add(ab, mu, o, ck, d)
        _add(ab, mu, o + 1, o + 1, -cc[i] * I); _add(ab, mu, o + 1, o + 2, -cc[i] * V)
        _add(ab, mu, o + 2, o + 1, -cc[i] * I); _add(ab, mu, o + 2, o + 2, -cc[i] * V)
        if prm[P_ann] > 0.0:
            for k in range(1, K + 1):
                ck = o + 2 + k
                dHV = ee[i] * I
                dI_ = ee[i] * y[ck]
                _add(ab, mu, ck, ck, -dHV); _add(ab, mu, ck, o + 2, -dI_)
                _add(ab, mu, o + 2, ck, -dHV); _add(ab, mu, o + 2, o + 2, -dI_)
                _add(ab, mu, o, ck, k * dHV); _add(ab, mu, o, o + 2, k * dI_)
        _add(ab, mu, o, o, -DHe[i] * 1e-12 * prm[P_SHe])
        _add(ab, mu, o + 1, o + 1, -DV[i] * 1e-12 * prm[P_SV])
        _add(ab, mu, o + 2, o + 2, -DI[i] * 1e-12 * prm[P_SI])
        if J > 0:
            kT = KB * T[i]
            k4 = 4.0 * np.pi * 1e-12 * prm[P_NC]
            sC0 = 3 + K
            kdim = k4 * prm[P_rHe1] * 2.0 * 2.0 * DHe[i]
            d_dim = kdim * He                                 # d rdim / dHe
            _add(ab, mu, o, o, -2.0 * d_dim)
            _add(ab, mu, o + sC0, o, d_dim)
            kb2 = prm[P_nuHeHe] * np.exp(-(EbC[0] + prm[P_EmHe]) / kT)
            _add(ab, mu, o + sC0, o + sC0, -kb2)
            _add(ab, mu, o, o + sC0, 2.0 * kb2)
            for jj in range(J):
                Cj = y[o + sC0 + jj]
                cj = o + sC0 + jj
                if jj + 2 <= K:
                    kv = k4 * prm[P_rc] * DV[i]
                    _add(ab, mu, cj, cj, -kv * V); _add(ab, mu, cj, o + 1, -kv * Cj)
                    _add(ab, mu, o + 1, cj, -kv * V); _add(ab, mu, o + 1, o + 1, -kv * Cj)
                    hv = o + 2 + (jj + 2)
                    _add(ab, mu, hv, cj, kv * V); _add(ab, mu, hv, o + 1, kv * Cj)
                if jj + 1 < J:
                    kg = k4 * rCl[jj] * DHe[i]
                    _add(ab, mu, cj, cj, -kg * He); _add(ab, mu, cj, o, -kg * Cj)
                    _add(ab, mu, o, cj, -kg * He); _add(ab, mu, o, o, -kg * Cj)
                    _add(ab, mu, cj + 1, cj, kg * He); _add(ab, mu, cj + 1, o, kg * Cj)
                    kb = prm[P_nuHeHe] * np.exp(-(EbC[jj + 1] + prm[P_EmHe]) / kT)
                    _add(ab, mu, cj + 1, cj + 1, -kb)
                    _add(ab, mu, cj, cj + 1, kb)
                    _add(ab, mu, o, cj + 1, kb)
                if nutm[jj] > 0.0 and jj + 2 <= K:
                    km = nutm[jj] * np.exp(-Etm[jj] / kT)
                    _add(ab, mu, cj, cj, -km)
                    _add(ab, mu, o + 2 + (jj + 2), cj, km)
                    _add(ab, mu, o + 2, cj, km)
