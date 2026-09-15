# -*- coding: utf-8 -*-
"""
Recursive Bayesian rows: the modified Cholesky with memory.

The ridge toward a prior is the posterior mode of a Gaussian regression, so
the posterior of one cycle is the prior of the next. Each row keeps the
sufficient statistics of its regression, ``G = X'X`` and ``b = X'y``,
initialised from the climatology and updated every cycle with the forecast
anomalies under a forgetting factor ``rho``:

    G_t = rho G_{t-1} + X_t' X_t,   b_t = rho b_{t-1} + X_t' y_t,
    beta_t = (G_t + a I)^{-1} b_t

so the coefficients of ``L`` are estimated from an effective sample of
``N / (1 - rho)`` members plus the climate, instead of ``N``. The structure
(which predecessors) is fixed, from the climatology; only the values learn.
``rho = 0`` recovers the plain per-cycle regression.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


class BayesRows:
    def __init__(self, grid, structure, clim_anom, alpha=0.3, rho=0.8, clim_weight=1.0, clim_decay=1.0, taper=0.0):
        """``clim_decay`` < 1: the climate prior fades on its own clock, w_c * decay^t,
        independently of the ensemble memory ``rho`` (1.0: the old behaviour)."""
        self.grid, self.n = grid, grid.n
        self.pred = [np.asarray(structure(i), dtype=int) for i in range(self.n)]
        self.alpha, self.rho = float(alpha), float(rho)
        self.clim_decay, self.t = float(clim_decay), 0
        self.taper = float(taper)
        self.TAPERS = (0.0, 0.1, 0.3, 0.5, 1.0)
        self.taper_auto = taper < 0
        if self.taper_auto:
            self.taper = 0.3
        self.taper_ll = None
        self.G, self.b, self.w = {}, {}, {}
        self.Gc, self.bc = {}, {}
        # tapered ridge: penalty of predecessor j grows as (1 + taper * d_ij)^2
        self.dist = [None] * self.n
        for i in np.flatnonzero(grid.interior):
            idx = self.pred[i]
            if idx.size:
                self.dist[i] = np.array([grid.chebyshev(int(i), int(j)) for j in idx], dtype=float)
                X = clim_anom[idx].T; y = clim_anom[i]
                Xs = X / (X.std(axis=0) + 1e-300); ys = y / (y.std() + 1e-300)   # correlation scale
                self.Gc[i] = clim_weight * (Xs.T @ Xs) / X.shape[0]; self.bc[i] = clim_weight * (Xs.T @ ys) / X.shape[0]
                if self.clim_decay >= 1.0:
                    self.G[i] = self.Gc[i].copy(); self.b[i] = self.bc[i].copy(); self.w[i] = clim_weight
                else:
                    self.G[i] = np.zeros_like(self.Gc[i]); self.b[i] = np.zeros_like(self.bc[i]); self.w[i] = 0.0
        self.last_wc = np.ones(self.n)

    def _pen(self, i, taper):
        return (1.0 + taper * self.dist[i]) ** 2

    def _row_ll(self, Xs, ys, Gp, bp, a_rel, pen, N):
        """Log marginal likelihood of the new anomalies under prior N(beta_prev, (N(Gp + a pen))^-1)."""
        p = Gp.shape[0]
        Pm = Gp + np.diag(a_rel * pen) + 1e-9 * np.eye(p)
        beta_prev = np.linalg.solve(Pm, bp)
        r = ys - Xs @ beta_prev
        M = np.eye(N) + Xs @ np.linalg.solve(N * Pm, Xs.T)
        cM = np.linalg.cholesky(M + 1e-9 * np.eye(N))
        z = np.linalg.solve(cM, r); s2 = float(z @ z) / N
        return -N / 2 * np.log(max(s2, 1e-300)) - np.log(np.diag(cM)).sum()

    RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)

    WCS = (0.0, 0.1, 0.3, 1.0)          # fractions of the initial climate weight, when chosen by evidence

    def _prior(self, i, wc=None):
        """Prior statistics for this cycle: ensemble memory plus the (fading or chosen) climate."""
        if self.clim_decay >= 1.0:
            return self.G[i], self.b[i]
        if wc is None:
            wc = self.clim_decay ** self.t if self.clim_decay > 0 else self.last_wc[i]
        return self.G[i] + wc * self.Gc[i], self.b[i] + wc * self.bc[i]

    def _evidence_rho_wc(self, Xs, ys, i, N):
        """Joint choice of forgetting factor and climate weight by marginal likelihood."""
        best, best_ll = (None, None), -np.inf
        for wc in self.WCS:
            Gp, bp = self.G[i] + wc * self.Gc[i], self.b[i] + wc * self.bc[i]
            p = Gp.shape[0]
            if not np.isfinite(Gp).all() or np.trace(Gp) < 1e-10:
                continue                                   # no prior information at all for this weight
            for rho in self.RHOS:
                Gr = rho * Gp + 1e-9 * np.eye(p)
                try:
                    beta_prev = np.linalg.solve(Gr, rho * bp)
                    r = ys - Xs @ beta_prev
                    Pinv = N * Gr
                    M = np.eye(N) + Xs @ np.linalg.solve(Pinv, Xs.T)
                    cM = np.linalg.cholesky(M + 1e-6 * np.eye(N))
                except np.linalg.LinAlgError:
                    continue
                z = np.linalg.solve(cM, r); s2 = float(z @ z) / N
                ll = -N / 2 * np.log(max(s2, 1e-300)) - np.log(np.diag(cM)).sum()
                if ll > best_ll:
                    best_ll, best = ll, (rho, wc)
        if best[0] is None:                                  # nothing to compare against: plain regression this cycle
            best = (0.0, 0.0)
        return best

    def _evidence_rho(self, Xs, ys, i, N):
        """Per-row forgetting factor maximising the marginal likelihood of the new anomalies."""
        Gp, bp = self._prior(i)
        p = Gp.shape[0]; Gr = Gp + 1e-9 * np.eye(p)
        beta_prev = np.linalg.solve(Gr, bp)
        r = ys - Xs @ beta_prev
        best, best_ll = None, -np.inf
        for rho in self.RHOS:
            Pinv = rho * N * Gr                              # prior precision of beta in data units
            M = np.eye(N) + Xs @ np.linalg.solve(Pinv, Xs.T)   # (N, N)
            cM = np.linalg.cholesky(M + 1e-9 * np.eye(N))
            z = np.linalg.solve(cM, r); s2 = float(z @ z) / N
            ll = -N / 2 * np.log(max(s2, 1e-300)) - np.log(np.diag(cM)).sum()
            if ll > best_ll:
                best_ll, best = ll, rho
        return best

    def update_and_build(self, DX):
        """Absorb one forecast ensemble and return B^-1 (sparse)."""
        rows, cols, vals = [], [], []; d = np.empty(self.n)
        N = DX.shape[1]
        self.last_rho = np.full(self.n, np.nan)
        if not hasattr(self, "last_wc"):
            self.last_wc = np.ones(self.n)
        taper_ll = np.zeros(len(self.TAPERS)) if self.taper_auto else None
        for i in range(self.n):
            rows.append(i); cols.append(i); vals.append(1.0)
            if not self.grid.interior[i]:
                d[i] = 1.0; continue
            idx = self.pred[i]; y = DX[i]
            if idx.size == 0:
                v = y.var(); d[i] = 1 / v if v > 0 else 0.0; continue
            X = DX[idx].T
            sx = X.std(axis=0) + 1e-300; sy = y.std() + 1e-300
            Xs = X / sx; ys = y / sy
            if self.clim_decay < 0:                       # rho and climate weight both by evidence
                rho, wc = self._evidence_rho_wc(Xs, ys, i, N); self.last_wc[i] = wc
                # the memory accumulates ensemble statistics only; the climate stays a
                # separate term whose weight is re-chosen every cycle and never absorbed
                self.G[i] = rho * self.G[i] + (Xs.T @ Xs) / N
                self.b[i] = rho * self.b[i] + (Xs.T @ ys) / N
                self.last_rho[i] = rho
                Gp, bp = self.G[i] + wc * self.Gc[i], self.b[i] + wc * self.bc[i]
            else:
                rho = self.rho if self.rho >= 0 else self._evidence_rho(Xs, ys, i, N)
                self.last_rho[i] = rho
                # accumulate per-sample correlation statistics: every cycle weighs one
                self.G[i] = rho * self.G[i] + (Xs.T @ Xs) / N
                self.b[i] = rho * self.b[i] + (Xs.T @ ys) / N
                self.w[i] = rho * self.w[i] + 1.0
                Gp, bp = self._prior(i)
            Gm = Gp.copy(); p = idx.size
            a_rel = self.alpha * np.trace(Gm) / p
            if self.taper_auto:
                rho_i = self.last_rho[i] if np.isfinite(self.last_rho[i]) else 0.8
                for ti, tp in enumerate(self.TAPERS):
                    taper_ll[ti] += self._row_ll(Xs, ys, rho_i * Gp, rho_i * bp, a_rel, self._pen(i, tp), N)
            Gm.flat[::p + 1] += a_rel * (self._pen(i, self.taper) if self.taper > 0 else 1.0)
            beta = np.linalg.solve(Gm, bp) * sy / sx           # back to the current scale
            v = float((y - X @ beta).var())          # residual variance on the current ensemble
            d[i] = 1 / v if v > 0 else 0.0
            rows.extend([i] * p); cols.extend(idx.tolist()); vals.extend((-beta).tolist())
        self.t += 1
        if self.taper_auto:
            self.taper_ll = taper_ll
            self.taper = self.TAPERS[int(np.argmax(taper_ll))]      # used from the next cycle on
        L = csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return (L.T @ diags(d, format="csr") @ L).tocsr()
