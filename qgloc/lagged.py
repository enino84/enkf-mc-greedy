# -*- coding: utf-8 -*-
"""
Predecessors from how the perturbations actually travelled: the ensemble as
its own perturbation experiment.

The analysis ensemble of the previous cycle is a set of N perturbations; the
forecast ensemble is the same perturbations propagated one interval. The
cross-correlation over members between the forecast anomaly at ``i`` and the
analysis anomaly at ``j`` estimates the propagator: from which points the
forecast at ``i`` inherited its error, with the direction and the lag the
dynamics gave it. Component ``i`` takes as predecessors the ``j`` within a
window whose lagged cross-correlation exceeds the sampling noise ``c/sqrt(N)``,
plus a square of radius ``local`` so the row always has its immediate
context. Recomputed every cycle; no extra propagation, no assumed velocity.
"""
from __future__ import annotations

import numpy as np


class LaggedGrid:
    def __init__(self, grid, DXa, DXf, window=6, c=2.0, local=1):
        self.base = grid
        self.n, self.g = grid.n, grid.g
        self.rows, self.cols, self.interior = grid.rows, grid.cols, grid.interior
        self.n_interior = grid.n_interior
        N = DXf.shape[1]
        thr = c / np.sqrt(N)
        A = DXa / (DXa.std(axis=1, keepdims=True) + 1e-300)
        F = DXf / (DXf.std(axis=1, keepdims=True) + 1e-300)
        self.pred = [np.empty(0, dtype=int)] * self.n
        self.support = np.zeros(self.n, dtype=int)
        for i in np.flatnonzero(grid.interior):
            nb = grid.neighbours(i, window, interior_only=True)
            cc = (F[i] @ A[nb].T) / (N - 1)
            keep = nb[np.abs(cc) > thr]
            if local > 0:
                keep = np.union1d(keep, grid.neighbours(i, local, interior_only=True))
            keep = keep[keep != i]
            self.support[i] = keep.size
            self.pred[i] = keep[keep < i]

    def predecessors(self, i, r=None):
        return self.pred[int(i)]

    def neighbours(self, i, r, interior_only=True):
        return self.base.neighbours(i, r, interior_only)

    def chebyshev(self, i, j):
        return self.base.chebyshev(i, j)
