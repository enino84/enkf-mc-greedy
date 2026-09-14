# -*- coding: utf-8 -*-
"""
B^{-1} from the modified Cholesky decomposition, one regression per component.

Component ``i`` is regressed on its predecessors ``P(i)``, the neighbours with
a lower index within its own radius ``r_i``. The coefficients fill row ``i`` of
a unit lower-triangular ``L`` and the residual variance fills ``D``, and
``B^{-1} = L' D^{-1} L``. The radius may differ per component: that is what
the assignment produces, and the row of ``L`` for ``i`` depends only on
``(i, r_i)``, so rows are cached by that pair.

The ridge is relative, not absolute
-----------------------------------
Each row solves ``(X'X + a_i I) beta = X'y`` with ``X`` the ``N x p_i`` matrix
of predecessor anomalies. An absolute ``a`` does not regularize anything
here: the entries of ``X'X`` scale with the variance of ``q`` and with ``N``,
so a fixed constant is either negligible or overwhelming depending on the
units, and it was measured to be negligible -- precisions nine orders of
magnitude too large, and a filter that ignored its observations. Worse, with
the assignment the number of predecessors ``p_i`` ranges from zero to more
than a hundred inside one cycle, so an absolute constant would regularize the
short rows hard and the wide rows not at all.

The penalty used is a fraction of the mean eigenvalue of the row's own Gram
matrix,

    a_i = alpha * trace(X'X) / p_i

which makes ``alpha`` dimensionless and gives it the same meaning in every
row, every cycle and every arm of the benchmark. The value ``alpha = 0.3``
is what converged in the cycled filter; the tables report the sensitivity.

The boundary
------------
Boundary components are not estimated. ``active_mask`` marks the interior;
inactive rows get a unit diagonal and no predecessors, which leaves their
analysis increment at zero -- correct for a component the Dirichlet condition
already fixes. Components with no ensemble variance are treated the same way
whatever the mask says, since ``1 / var`` would be infinite.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


def ridge_scaled(X, y, alpha):
    """``(X'X + a I)^{-1} X'y`` with ``a = alpha * trace(X'X) / p``."""
    p = X.shape[1]
    G = X.T @ X
    G.flat[:: p + 1] += alpha * (np.trace(G) / p + 1e-300)
    return np.linalg.solve(G, X.T @ y)


class PrecisionBuilder:
    """``B^{-1}`` for a per-component radius field, rows cached by (i, r)."""

    def __init__(self, grid, alpha=0.3, active_mask=None, var_floor=0.0):
        self.grid = grid
        self.n = int(grid.n)
        self.alpha = float(alpha)
        self.var_floor = float(var_floor)
        self.active = (grid.interior.copy() if active_mask is None
                       else np.asarray(active_mask, dtype=bool).copy())
        self._rows = {}
        self._DX_id = None
        self.n_row_builds = 0
        self.n_row_hits = 0

    # ------------------------------------------------------------------
    def bind(self, DX):
        """Attach an anomaly matrix ``(n, N)``; a new one clears the cache."""
        key = id(DX)
        if key != self._DX_id:
            self._rows.clear()
            self._DX_id = key
        self.DX = np.asarray(DX, dtype=float)
        self.N = self.DX.shape[1]
        zero_var = self.DX.var(axis=1) <= 0.0
        self.active = self.active & ~zero_var
        return self

    def row(self, i, r):
        """Predecessors, coefficients and ``1/d_i`` for component ``i``."""
        key = (int(i), int(r))
        cached = self._rows.get(key)
        if cached is not None:
            self.n_row_hits += 1
            return cached
        self.n_row_builds += 1

        if not self.active[i]:
            out = (np.empty(0, dtype=int), np.empty(0), 1.0)
            self._rows[key] = out
            return out

        y = self.DX[i, :]
        idx = self.grid.predecessors(i, r) if r > 0 else np.empty(0, dtype=int)
        if idx.size:
            idx = idx[self.active[idx]]
        if idx.size == 0:
            v = float(np.var(y))
            out = (idx, np.empty(0), 1.0 / v if v > 0 else 0.0)
        else:
            X = self.DX[idx, :].T
            beta = ridge_scaled(X, y, self.alpha)
            v = float(np.var(y - X @ beta))
            v = max(v, self.var_floor * float(np.var(y)))
            out = (idx, beta, 1.0 / v if v > 0 else 0.0)
        self._rows[key] = out
        return out

    # ------------------------------------------------------------------
    def build(self, r_field):
        """Sparse ``B^{-1}`` for one radius per component."""
        r = np.asarray(r_field)
        if r.ndim == 0:
            r = np.full(self.n, int(r))
        r = np.floor(r).astype(int)

        rows, cols, vals = [], [], []
        d = np.empty(self.n)
        for i in range(self.n):
            idx, beta, di = self.row(i, r[i])
            d[i] = di
            rows.append(i); cols.append(i); vals.append(1.0)
            if idx.size:
                rows.extend([i] * idx.size)
                cols.extend(idx.tolist())
                vals.extend((-beta).tolist())
        L = csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return (L.T @ diags(d, format="csr") @ L).tocsr()

    def structure(self, r_field):
        """Only the sparsity: number of nonzeros of ``B^{-1}`` and of ``L``."""
        B = self.build(r_field)
        n_pred = sum(self.row(i, int(r)).__getitem__(0).size
                     for i, r in enumerate(np.floor(np.asarray(r_field)).astype(int)))
        return dict(nnz=int(B.nnz), density=float(B.nnz) / self.n ** 2,
                    n_predecessors=int(n_pred))

    def stats(self):
        return dict(cached_rows=len(self._rows), builds=self.n_row_builds,
                    hits=self.n_row_hits)
