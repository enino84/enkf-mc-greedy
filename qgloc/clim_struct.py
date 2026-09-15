# -*- coding: utf-8 -*-
"""
Structure from the climatology, values from the ensemble.

Every structure read from a 40-member ensemble pays selection noise and ends
up between uniform r = 1 and r = 2. The climatology has hundreds of states and
no such noise. Here the predecessor sets are chosen once, from a lasso row
regression on the climatological anomalies over a wide window, and kept fixed
for the whole run; each cycle the modified Cholesky refits the coefficients on
the forecast ensemble as usual. If the QG has a stationary conditional
structure richer than a square (the jet's bands), this is where it shows
clean; if this does not beat the uniform radius, no structure will.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso


class ClimGrid:
    def __init__(self, grid, clim_q, window=4, c=1.0, local=1, max_iter=1000):
        """``clim_q``: (n, M) climatological states of q (M snapshots)."""
        self.base = grid
        self.n, self.g = grid.n, grid.g
        self.rows, self.cols, self.interior = grid.rows, grid.cols, grid.interior
        self.n_interior = grid.n_interior
        A = clim_q - clim_q.mean(axis=1, keepdims=True)
        M = A.shape[1]
        Z = A / (A.std(axis=1)[:, None] + 1e-300)
        self.pred = [np.empty(0, dtype=int)] * self.n
        self.support = np.zeros(self.n, dtype=int)
        for i in np.flatnonzero(grid.interior):
            cand = grid.neighbours(i, window, interior_only=True)
            cand = cand[cand < i]
            if cand.size == 0:
                continue
            lam = c * np.sqrt(2.0 * np.log(max(cand.size, 2)) / M)
            m = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter, tol=1e-4).fit(Z[cand].T, Z[i])
            keep = cand[np.abs(m.coef_) > 1e-8]
            if local > 0:
                loc = grid.neighbours(i, local, interior_only=True); keep = np.union1d(keep, loc[loc < i])
            self.support[i] = keep.size
            self.pred[i] = keep

    def predecessors(self, i, r=None):
        return self.pred[int(i)]

    def neighbours(self, i, r, interior_only=True):
        return self.base.neighbours(i, r, interior_only)

    def chebyshev(self, i, j):
        return self.base.chebyshev(i, j)
