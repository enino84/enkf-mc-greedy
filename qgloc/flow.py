# -*- coding: utf-8 -*-
"""
The predecessor structure of the modified Cholesky read from the flow.

The modified Cholesky needs an order of the variables and, for each, a set of
predecessors. The standard choice is the lexicographic order of the grid and
a square of radius ``r``. Here both are taken from the dynamics:

* The velocity field ``(u, v) = (-dpsi/dy, dpsi/dx)`` of the ensemble mean,
  which the model already uses to advect ``q``.
* The **wake** of a component: the grid cells within ``width`` of its
  backward trajectory over one assimilation interval, capped at ``cap`` grid
  points of travel. Those are the points from which the flow could have
  brought background error to it during the forecast.
* The **order**: upstream first, by an approximate topological sort of the
  wake graph (flow depth). The predecessors of a component are the members
  of its wake that precede it in that order, so ``L'DL`` is a valid
  precision. Closed streamlines create cycles; the depth cap breaks them.

There is no radius. The wake length is set by the assimilation interval and
the speed of the flow; ``cap`` only keeps a row from having more predecessors
than ``N`` members can determine, and ``width`` gives the regression one cell
of cross-stream context. Both are fixed, not tuned.

Cost: one backward integration of all grid points (vectorised, 200 steps of
RK2 with bilinear interpolation), a few sweeps for the order, then the usual
row regressions. All from the ensemble; the observation values do not enter.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def velocity_from_psi(psi):
    """``u = -dpsi/dy``, ``v = dpsi/dx`` on the grid (rows = y, cols = x)."""
    v = np.gradient(psi, axis=1)
    u = -np.gradient(psi, axis=0)
    return u, v


def backward_trajectories(u, v, T, cap, nsteps=200, points=None):
    """Backward RK2 trajectories of every grid point over time ``T``.

    Stops a trajectory once it has travelled ``cap`` grid points. Returns an
    array ``(nsteps + 1, 2, n)`` of (row, col) positions.
    """
    g = u.shape[0]
    n = g * g if points is None else len(points)
    P = (np.stack(np.divmod(np.arange(g * g), g)) if points is None
         else np.stack(np.divmod(np.asarray(points), g))).astype(float)

    def vel(Q):
        return np.stack([map_coordinates(u, Q, order=1, mode="nearest"),
                         map_coordinates(v, Q, order=1, mode="nearest")])

    dt = T / nsteps
    out = [P.copy()]
    length = np.zeros(n)
    for _ in range(nsteps):
        Q = out[-1]
        k1 = vel(Q)
        k2 = vel(np.clip(Q - 0.5 * dt * k1, 0, g - 1))
        step = dt * k2
        alive = length < cap
        Qn = np.where(alive, np.clip(Q - step, 0, g - 1), Q)
        length += np.where(alive, np.hypot(step[0], step[1]), 0.0)
        out.append(Qn)
    return np.array(out)


class FlowGrid:
    """Predecessor sets and order from the flow; drop-in for ``geometry.Grid``.

    Built once per cycle from the ensemble mean. ``predecessors(i, r)`` ignores
    ``r``: the structure has no radius.
    """

    def __init__(self, grid, psi, T, cap=8.0, width=1.0, nsteps=200, max_depth=200, local=0):
        """``local``: radius of a square neighbourhood added to every wake, so the
        structure keeps the local, isotropic part the wake alone does not carry."""
        self.local = int(local)
        self.base = grid
        self.n, self.g = grid.n, grid.g
        self.rows, self.cols, self.interior = grid.rows, grid.cols, grid.interior
        self.n_interior = grid.n_interior
        self.cap, self.width = float(cap), float(width)
        u, v = velocity_from_psi(np.asarray(psi).reshape(self.g, self.g))
        self.u, self.v = u, v
        self.trajs = backward_trajectories(u, v, T, cap, nsteps=nsteps)
        self._build_wakes()
        self._order(max_depth)

    def _build_wakes(self):
        n, g = self.n, self.g
        cr, cc = self.rows, self.cols
        sub = self.trajs[::10]                       # (k, 2, n) polyline samples
        self.wake = [np.empty(0, dtype=int)] * n
        for i in np.flatnonzero(self.interior):
            tr = sub[:, :, i]                        # (k, 2)
            # cells near the polyline: bound the search to the trajectory's box
            r0, r1 = int(tr[:, 0].min() - self.width), int(np.ceil(tr[:, 0].max() + self.width))
            c0, c1 = int(tr[:, 1].min() - self.width), int(np.ceil(tr[:, 1].max() + self.width))
            rr, cc2 = np.meshgrid(np.arange(max(0, r0), min(g, r1 + 1)),
                                  np.arange(max(0, c0), min(g, c1 + 1)), indexing="ij")
            cand = (rr * g + cc2).ravel()
            d = np.min(np.hypot(cr[cand][:, None] - tr[None, :, 0],
                                cc[cand][:, None] - tr[None, :, 1]), axis=1)
            w = cand[(d <= self.width) & self.interior[cand]]
            if self.local > 0:
                w = np.union1d(w, self.base.neighbours(i, self.local, interior_only=True))
            self.wake[i] = w[w != i]

    def _order(self, max_depth):
        n = self.n
        depth = np.zeros(n)
        inter = np.flatnonzero(self.interior)
        for _ in range(max_depth // 4):
            new = depth.copy()
            for i in inter:
                w = self.wake[i]
                if w.size:
                    new[i] = min(max_depth, depth[w].max() + 1)
            if np.array_equal(new, depth):
                break
            depth = new
        self.depth = depth
        order = np.lexsort((np.arange(n), depth))
        self.pos = np.empty(n, dtype=int)
        self.pos[order] = np.arange(n)
        self.pred = [None] * n
        for i in range(n):
            w = self.wake[i]
            self.pred[i] = w[self.pos[w] < self.pos[i]] if w.size else w

    # -- Grid interface -------------------------------------------------
    def predecessors(self, i, r=None):
        return self.pred[int(i)]

    def neighbours(self, i, r, interior_only=True):
        return self.base.neighbours(i, r, interior_only)

    def chebyshev(self, i, j):
        return self.base.chebyshev(i, j)

    def n_predecessors(self):
        return np.array([p.size for p in self.pred])

    def wake_length(self):
        return np.array([w.size for w in self.wake])
