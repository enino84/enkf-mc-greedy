# -*- coding: utf-8 -*-
"""
The grid, its interior, and the square neighbourhoods on it.

Both analyses in this repository -- the modified-Cholesky EnKF and the LETKF
-- localize on the same geometry: a square (Chebyshev) neighbourhood of
integer radius around each grid point, **not wrapped** at the edges. This
module is the single place where that geometry is defined, so that a radius
means the same thing in the two filters and in the assignment.

Why not the model's own ``get_ngb``
-----------------------------------
pyteda's ``QGModel.get_ngb`` wraps the indices modulo the grid, which is right
for a doubly periodic domain and wrong here: the testbed is Dirichlet, ``q``
vanishes on the boundary, and a component on the left edge has nothing to do
with one on the right. Using the periodic neighbourhoods would give the
components near the boundary predecessors on the opposite side of the domain,
which are uncorrelated with them and merely spend regression degrees of
freedom. The neighbourhoods here stop at the edge.

The boundary is not estimated
-----------------------------
The ``g x g`` grid has ``(g-2)^2`` interior points. ``q`` is zero on the
boundary by construction, every ensemble member is zero there, and the
analysis increment is zero there whatever is done. Following Sakov and Oke,
who report the QG dimension excluding the boundary, the boundary components
are removed from the assimilation entirely: they are not observed, they are
not candidates, they are not predecessors, and they do not enter the error.
``interior`` is the mask; everything else in the package is written on the
full ``g x g`` index set with that mask applied.
"""
from __future__ import annotations

import numpy as np


class Grid:
    """A ``g x g`` grid with its interior mask and square neighbourhoods."""

    def __init__(self, g: int):
        self.g = int(g)
        self.n = self.g * self.g
        idx = np.arange(self.n)
        self.rows, self.cols = np.divmod(idx, self.g)
        m = np.ones((self.g, self.g), dtype=bool)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
        self.interior = m.ravel()
        self.n_interior = int(self.interior.sum())
        self._ngb = {}

    # ------------------------------------------------------------------
    def neighbours(self, i: int, r: int, interior_only: bool = True):
        """Indices within Chebyshev distance ``r`` of ``i``, edges not wrapped.

        Includes ``i`` itself. With ``interior_only`` the boundary is dropped,
        which is what both filters use.
        """
        key = (int(i), int(r), bool(interior_only))
        out = self._ngb.get(key)
        if out is not None:
            return out
        r = int(r)
        r0, c0 = int(self.rows[i]), int(self.cols[i])
        lo_r, hi_r = max(0, r0 - r), min(self.g - 1, r0 + r)
        lo_c, hi_c = max(0, c0 - r), min(self.g - 1, c0 + r)
        rr, cc = np.meshgrid(np.arange(lo_r, hi_r + 1),
                             np.arange(lo_c, hi_c + 1), indexing="ij")
        out = (rr * self.g + cc).ravel()
        if interior_only:
            out = out[self.interior[out]]
        out = np.sort(out)
        self._ngb[key] = out
        return out

    def predecessors(self, i: int, r: int):
        """Neighbours of ``i`` with index strictly below ``i``: the Cholesky set."""
        nb = self.neighbours(i, r, interior_only=True)
        return nb[nb < i]

    def chebyshev(self, i, j):
        return int(max(abs(self.rows[i] - self.rows[j]),
                       abs(self.cols[i] - self.cols[j])))

    def to_2d(self, v, fill=np.nan):
        """A full-grid vector as a ``g x g`` array; ``fill`` on the boundary."""
        a = np.full(self.n, fill, dtype=float)
        v = np.asarray(v)
        if v.size == self.n:
            a[:] = v
        elif v.size == self.n_interior:
            a[self.interior] = v
        else:
            raise ValueError("vector must be full-grid or interior-sized")
        return a.reshape(self.g, self.g)
