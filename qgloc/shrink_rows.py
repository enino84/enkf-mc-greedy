# -*- coding: utf-8 -*-
"""
Rows shrunk toward the climatology, with the strength chosen by the data.

Structure: the radius-1 square (always) plus the climatological predecessors
of ``ClimGrid``. Coefficients: ridge toward the climatological coefficients
``beta_c`` of the same row,

    beta = beta_c + (X'X + a I)^{-1} X' (y - X beta_c),

with ``a`` chosen per row and per cycle by leave-one-member-out cross
validation over a grid of relative strengths. When the ensemble error looks
like the climate (the first analysis) the data pick a large ``a``; once the
filter has converged they pick a small one. Nothing is tuned by hand.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


class ShrinkRows:
    def __init__(self, grid, clim_grid, clim_anom, alpha_clim=0.3):
        self.grid, self.n = grid, grid.n
        self.pred = clim_grid.pred                     # already includes the radius-1 square
        self.beta_c = {}
        for i in np.flatnonzero(grid.interior):
            idx = self.pred[i]
            if idx.size:
                X = clim_anom[idx].T; y = clim_anom[i]; G = X.T @ X; p = idx.size
                G.flat[::p + 1] += alpha_clim * np.trace(G) / p
                self.beta_c[i] = np.linalg.solve(G, X.T @ y)
        self.last_alpha = np.full(self.n, np.nan)

    def build(self, DX, alpha=None):
        """``alpha`` given: use it for every row (no cross-validation)."""
        rows, cols, vals = [], [], []; d = np.empty(self.n)
        N = DX.shape[1]
        for i in range(self.n):
            rows.append(i); cols.append(i); vals.append(1.0)
            if not self.grid.interior[i]:
                d[i] = 1.0; continue
            idx = self.pred[i]; y = DX[i]
            if idx.size == 0:
                v = y.var(); d[i] = 1 / v if v > 0 else 0.0; continue
            X = DX[idx].T; b0 = self.beta_c[i]; r = y - X @ b0          # residual of the climate coefficients
            G = X.T @ X; p = idx.size; tr = np.trace(G) / p
            best, best_loo = alpha, np.inf
            # LOO for ridge: e_loo = e / (1 - h_ii), h = X (X'X + aI)^-1 X'
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            for al in (ALPHAS if alpha is None else ()):
                a = al * tr
                shrink = s ** 2 / (s ** 2 + a)
                h = (U ** 2 * shrink).sum(axis=1)
                fit = U @ (shrink * (U.T @ r))
                loo = np.mean(((r - fit) / np.maximum(1 - h, 1e-6)) ** 2)
                if loo < best_loo:
                    best_loo, best = loo, al
            a = best * tr; Gm = G.copy(); Gm.flat[::p + 1] += a
            beta = b0 + np.linalg.solve(Gm, X.T @ r)
            self.last_alpha[i] = best
            v = float((y - X @ beta).var()); d[i] = 1 / v if v > 0 else 0.0
            rows.extend([i] * p); cols.extend(idx.tolist()); vals.extend((-beta).tolist())
        L = csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return (L.T @ diags(d, format="csr") @ L).tocsr()
