# -*- coding: utf-8 -*-
"""
Lorenz-96 with the same interface as the QG testbed: the sanity check.

Everything in ``filters.py``, ``precision.py`` and ``assignment.py`` runs
unchanged on this testbed; only the geometry differs (a periodic ring). If the
fixed-radius EnKF-MC does not behave here the way every localized EnKF does
on Lorenz-96 -- best radius around 2-4 at N = 20, error a fraction of the
climatological spread -- the problem is in the filter, not in the QG.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyteda.models.lorenz96 import Lorenz96


class Ring:
    """A periodic 1-D grid; every point is interior."""

    def __init__(self, n):
        self.n = int(n); self.g = self.n
        self.rows = np.zeros(self.n, dtype=int); self.cols = np.arange(self.n)
        self.interior = np.ones(self.n, dtype=bool)
        self.n_interior = self.n
        self._ngb = {}

    def neighbours(self, i, r, interior_only=True):
        key = (int(i), int(r))
        out = self._ngb.get(key)
        if out is None:
            out = np.sort(np.arange(i - int(r), i + int(r) + 1) % self.n)
            self._ngb[key] = out
        return out

    def predecessors(self, i, r):
        nb = self.neighbours(i, r)
        return nb[nb < i]

    def chebyshev(self, i, j):
        d = abs(int(i) - int(j))
        return min(d, self.n - d)


@dataclass
class L96Config:
    n: int = 40
    F: float = 8.0
    dt: float = 0.05
    obs_freq: float = 0.05         # one model step between analyses (standard)
    obs_std_raw: float = 1.0       # standard: obs error 1.0 in model units
    ensemble_size: int = 20
    inflation: float = 1.02
    obs_stride: int = 2
    obs_network: str = "lattice"
    obs_density: float = 0.5
    ridge_alpha: float = 0.3
    rho: int = 4
    r_cap: int | None = None
    kappa: int = 4
    r_max: int = 12
    cycles: int = 200
    burn_in: int = 50
    halo: int = 2

    @property
    def obs_std(self):
        return self.obs_std_raw / self._spread


class L96Testbed:
    def __init__(self, cfg: L96Config, spinup=100.0, n_snapshots=400, snapshot_every=0.5):
        self.cfg = cfg
        self.model = Lorenz96(n=cfg.n, F=cfg.F)
        self.n = self.nq = cfg.n
        self.g = cfg.n
        self.qblock = slice(0, self.n)
        self.blocks = {"q": self.qblock}
        self.grid = Ring(self.n)
        rng = np.random.default_rng(0)
        x = self.model.propagate(rng.standard_normal(self.n), np.array([0.0, spinup]))
        snaps = [x]
        for _ in range(n_snapshots):
            x = self.model.propagate(x, np.array([0.0, snapshot_every])); snaps.append(x)
        self.clim = np.array(snaps)
        self.spread = {"q": float(self.clim.std())}
        cfg._spread = self.spread["q"]

    def build_ensemble(self, seed):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.clim), size=self.cfg.ensemble_size + 1, replace=False)
        return self.clim[idx[1:]].T.copy(), self.clim[idx[0]].copy()

    def normalize(self, x):
        return x / self.spread["q"]

    def denormalize(self, x):
        return x * self.spread["q"]

    def psi_from_q(self, x):
        return x

    def lattice(self, stride=None, offset=0):
        return np.arange(int(offset), self.n, int(stride or self.cfg.obs_stride))

    def network(self, cycle=0, seed=0, kind=None, stride=None):
        kind = kind or self.cfg.obs_network
        if kind == "lattice":
            return self.lattice(stride)
        count = int(round(self.cfg.obs_density * self.n))
        s = int(seed) * 100003 + (1 + int(cycle) if kind == "random-moving" else 0)
        return np.sort(np.random.default_rng(s).choice(self.n, size=count, replace=False))

    def errors(self, x, x_true):
        d = (x - x_true) / self.spread["q"]
        r = float(np.sqrt(np.mean(d ** 2)))
        return dict(rmse_q=r, rmse_q_raw=r * self.spread["q"], rel_q=r, rmse_psi=np.nan, rmse=r)
