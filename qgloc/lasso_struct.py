# -*- coding: utf-8 -*-
"""
Predecessors chosen by the regression itself: an L1-penalised row.

For every component the candidates are its neighbours with a lower index
within a wide window. A lasso regression of its forecast anomaly on theirs
sets to zero the coefficients that do not earn their place; the survivors are
the predecessors, and the row of ``L`` is refit by ridge on that support (a
relaxed lasso, so the selection does not bias the values). The penalty is the
universal threshold ``lam = c * sigma_y * sqrt(2 ln p / N)`` with ``c`` the
one dimensionless parameter, tied to the sampling noise like the threshold
rule but applied jointly to the whole row rather than pair by pair.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso


class LassoGrid:
    def __init__(self, grid, DXf, window=4, c=1.0, local=1, max_iter=500):
        self.base = grid
        self.n, self.g = grid.n, grid.g
        self.rows, self.cols, self.interior = grid.rows, grid.cols, grid.interior
        self.n_interior = grid.n_interior
        N = DXf.shape[1]
        sd = DXf.std(axis=1) + 1e-300
        Z = DXf / sd[:, None]
        self.pred = [np.empty(0, dtype=int)] * self.n
        self.support = np.zeros(self.n, dtype=int)
        for i in np.flatnonzero(grid.interior):
            cand = grid.neighbours(i, window, interior_only=True)
            cand = cand[cand < i]
            if cand.size == 0:
                continue
            X = Z[cand].T; y = Z[i]
            p = cand.size
            lam = c * np.sqrt(2.0 * np.log(max(p, 2)) / N)
            m = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter, tol=1e-4)
            m.fit(X, y)
            keep = cand[np.abs(m.coef_) > 1e-8]
            if local > 0:
                loc = grid.neighbours(i, local, interior_only=True)
                keep = np.union1d(keep, loc[loc < i])
            self.support[i] = keep.size
            self.pred[i] = keep

    def predecessors(self, i, r=None):
        return self.pred[int(i)]

    def neighbours(self, i, r, interior_only=True):
        return self.base.neighbours(i, r, interior_only)

    def chebyshev(self, i, j):
        return self.base.chebyshev(i, j)
