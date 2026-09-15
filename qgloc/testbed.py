# -*- coding: utf-8 -*-
"""
The quasi-geostrophic testbed: model, ensemble, observations.

Three things in this module are decisions rather than conveniences, and each
was arrived at by measuring what happened without it.

**Only ``q`` is estimated.** The model integrates ``q`` and recovers ``psi``
by solving the Helmholtz problem at every step, so ``psi`` is a diagnostic and
not a prognostic variable: ``pyteda``'s ``propagate`` reads ``x0[:field_size]``
and calls ``_calc_psi(q)``, ignoring whatever ``psi`` it was handed. Measured
directly: halving ``psi`` while leaving ``q`` untouched and propagating for 20
time units gives a bit-identical result.

An analysis that corrects the ``psi`` block therefore throws that correction
away at the next propagation. The state estimated here is ``q`` alone, 2401
components rather than 4802, and ``psi`` is recomputed from the analysed ``q``.
That halves the number of regressions, halves the dimension of the linear
solve, and takes the radius vector from ``2K`` components to ``K``.

**The state is normalized per field.** The potential vorticity ``q`` has a
climatological standard deviation of order 2000 and the streamfunction ``psi``
of order 1. An observation error that is reasonable for one is meaningless for
the other, and pyteda's LETKF reads a single scalar variance off the noise
object, so a heterogeneous R silently applies the first field's error to both.
Dividing each field by its climatological spread turns that into a
non-problem: in normalized units both fields have unit spread, one isotropic
error is correct for both, and the aggregate RMSE stops being dominated by
``q`` by four orders of magnitude.

**Three observation networks, all on the interior.** A fixed regular lattice
of spacing ``stride``, which fixes the distance from every point to its
nearest observation; a random network observing a fraction ``obs_density`` of
the interior, drawn once and kept for the whole run, where that distance
varies in space but not in time; and a random network of the same fraction
redrawn at every assimilation step, where it varies in both.
The lattice is where the relation between density and radius is cleanest; the
random networks are where a radius that adapts to the local density has
something to adapt to. None places observations on the Dirichlet boundary,
where ``q`` is identically zero: see ``geometry.py``.

**The ensemble is climatological, not perturbed.** This is the recipe of Sakov
and Oke (2008): run the model once for a long time, keep snapshots along the
way, and draw the members and the truth at random from that set. Perturbing a
state and propagating does not work here, and the reason is measurable. A
perturbation is mostly energy outside the attractor, and the hyperviscosity
removes it: starting from 30% of climatology, the background error falls to
0.068 after 120 time units, a factor of five, and no amount of raising the
perturbation amplitude compensates because the propagation eats it. Only after
about 250 units does the surviving component start to grow, reaching 0.872 at
t = 1000 -- which is exactly the 0.873 a climatological ensemble has from the
start. The long propagation is a slow way of arriving where the snapshots
already are.

**The dissipation is ten times the model default.** With ``rkh2 = 1e-12`` the
norm of ``q`` grows without saturating -- a factor of fourteen between t = 500
and t = 8000 -- so there is no stationary regime and no climatology to sample.
At ``rkh2 = 1e-11`` it settles: 1.107e4 at t = 20000 against 1.222e4 at
t = 60000. Sakov and Oke report the same, raising the dissipation by a factor
of ten and reducing the step from 1.5 to 1.25 to get a stable assimilating
system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyteda.models import QGModel


# ----------------------------------------------------------------------
@dataclass
class QGConfig:
    """Everything that defines the testbed."""
    mrefin: int = 6              # 6 -> 97x97 per field, 7 -> 193x193
    # Sakov and Oke use 1.25 for the assimilating run, arrived at by reducing
    # it from the 1.5 that free runs allow. The same reduction is needed again
    # here and for the same reason they give: the corrections localization
    # introduces are dynamically inconsistent, and the biharmonic term
    # amplifies the short scales they create until the forecast blows up. The
    # failure is in `propagate`, not in the analysis.
    dt: float = 0.3125
    bc: str = "dirichlet"
    scheme: str = "rk4"
    rkh2: float = 1e-11          # ten times the model default; see the header
    # Climatology: one long run, snapshots along it, members drawn at random.
    spinup: float = 20000.0      # to the stationary regime, cached
    n_snapshots: int = 60
    snapshot_every: float = 250.0
    ensemble_size: int = 20
    obs_stride: int = 4          # lattice spacing; density is 1/stride^2
    obs_network: str = "lattice" # lattice | random-fixed | random-moving
    obs_density: float = 0.25    # fraction of the interior observed by the random networks
    obs_std: float = 0.05        # in normalized units, so 5% of each spread
    obs_freq: float = 20.0       # time between assimilation cycles
    inflation: float = 1.05         # 1.15 blows the spread up over 60 cycles on a fixed lattice, measured
    cycles: int = 60
    burn_in: int = 15
    # Modified-Cholesky ridge, as a fraction of the mean eigenvalue of each
    # row's Gram matrix (see precision.py). Not an absolute constant.
    ridge_alpha: float = 0.3
    # Assignment: candidates per point and the cap on how far a neighbourhood
    # may grow while collecting them.
    kappa: int = 4
    r_max: int = 12
    # EnKF-MC-group: width of the halo around each group.
    halo: int = 2
    # The method: probe radius of the partial-correlation rule, and an
    # optional cap on the assigned radii (None: up to 2*rho can occur).
    rho: int = 4
    r_cap: int | None = None
    # EnKF-MC-flow: wake length cap (grid points of travel) and half-width.
    wake_cap: float = 8.0
    wake_width: float = 1.0
    wake_local: int = 2             # square of this radius added to every wake (2 measured best at N = 40)


def build_model(cfg: QGConfig) -> QGModel:
    return QGModel(mrefin=cfg.mrefin, scheme=cfg.scheme, dt=cfg.dt,
                   bc=cfg.bc, rkh2=cfg.rkh2, ic_kind="zero", verbose=False)


class Testbed:
    """Model, blocks, climatological scales and the normalization they define."""

    def __init__(self, cfg: QGConfig, snapshots: np.ndarray):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.n = self.model.get_number_of_variables()
        self.blocks = dict(self.model.var_blocks)
        self.g = int(np.sqrt(self.blocks["q"].stop - self.blocks["q"].start))
        self.snapshots = np.asarray(snapshots, dtype=float)
        if self.snapshots.ndim != 2 or self.snapshots.shape[1] != self.n:
            raise ValueError(f"snapshots must be (n_snapshots, {self.n})")
        self.x0_ref = self.snapshots[0]

        # Climatological spread per field, over the whole snapshot set rather
        # than one state, since that is what "climatological" means and what
        # the normalization should be expressed in.
        self.spread = {k: float(self.snapshots[:, b].std())
                       for k, b in self.blocks.items()}
        self.scale = np.ones(self.n)
        for k, b in self.blocks.items():
            self.scale[b] = self.spread[k]

    # ------------------------------------------------------------------
    @property
    def qblock(self):
        """The prognostic block. Everything estimated lives here."""
        return self.blocks["q"]

    @property
    def nq(self):
        return self.qblock.stop - self.qblock.start

    def psi_from_q(self, x):
        """Recompute psi from q, the way the model does at every step.

        Used after an analysis so that the state handed back satisfies
        Lap psi - F psi = q, which an independently corrected psi would not.
        """
        x = np.asarray(x, dtype=float)
        out = x.copy()
        b = self.blocks["psi"]
        core = getattr(self.model, "_core", None)
        q2 = x[self.qblock].reshape(self.g, self.g)
        if core is not None and hasattr(core, "_calc_psi"):
            out[b] = np.asarray(core._calc_psi(q2)).ravel()
        return out

    def normalize(self, x):
        return np.asarray(x) / (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def denormalize(self, x):
        return np.asarray(x) * (self.scale if np.ndim(x) == 1
                                else self.scale[:, None])

    def boundary_mask(self):
        """True on the grid points that lie on the Dirichlet boundary."""
        g = self.g
        m2 = np.zeros((g, g), dtype=bool)
        m2[0, :] = m2[-1, :] = m2[:, 0] = m2[:, -1] = True
        m = np.zeros(self.n, dtype=bool)
        for b in self.blocks.values():
            m[b] = m2.ravel()
        return m

    # ------------------------------------------------------------------
    def build_ensemble(self, seed):
        """Members and truth drawn at random from the climatological set.

        No perturbation and no per-member spin-up: every member is already a
        state of the attractor, and the dispersion between them is the
        variability the model itself produces. This is what Sakov and Oke do,
        and the measurements in the module header say why nothing else works.
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        need = cfg.ensemble_size + 1
        if self.snapshots.shape[0] < need:
            raise ValueError(f"need {need} snapshots, have "
                             f"{self.snapshots.shape[0]}")
        pick = rng.choice(self.snapshots.shape[0], size=need, replace=False)
        x_true = self.snapshots[pick[0]].copy()
        X = self.snapshots[pick[1:]].T.copy()
        return X, x_true

    # ------------------------------------------------------------------
    @property
    def grid(self):
        if getattr(self, "_grid", None) is None:
            from .geometry import Grid
            self._grid = Grid(self.g)
        return self._grid

    def lattice(self, stride=None, offset=0):
        """Interior q-indices on a regular lattice of the given spacing.

        A fixed count of evenly spaced positions over the interior rows and
        columns, so that shifting the lattice by ``offset`` moves the network
        without changing how many observations it has. Boundary rows and
        columns are never used.
        """
        g = self.g
        stride = int(stride or self.cfg.obs_stride)
        inner = g - 2                      # interior side
        k = max(1, inner // stride)
        base = np.round(np.linspace(0, inner - inner / k, k)).astype(int)
        pos = 1 + (base + int(offset)) % inner      # 1 .. g-2
        rows = np.zeros(g, dtype=bool); rows[np.unique(pos)] = True
        mask = np.outer(rows, rows).ravel() & self.grid.interior
        return np.flatnonzero(mask)

    def random_network(self, count, seed):
        """``count`` interior q-indices drawn uniformly without replacement."""
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(self.grid.interior)
        return np.sort(rng.choice(pool, size=int(count), replace=False))

    def network(self, cycle=0, seed=0, kind=None, stride=None):
        """The observation network of one cycle, as interior q-indices.

        Three kinds:
          ``lattice``        a fixed regular lattice of spacing ``obs_stride``
          ``random-fixed``   ``obs_density`` of the interior, drawn once per
                             seed and kept for the whole run
          ``random-moving``  the same count, redrawn at every assimilation step
        """
        kind = kind or self.cfg.obs_network
        if kind == "lattice":
            return self.lattice(stride=stride, offset=0)
        count = int(round(float(self.cfg.obs_density) * self.grid.n_interior))
        if kind == "random-fixed":
            return self.random_network(count, seed=int(seed) * 100003)
        if kind == "random-moving":
            return self.random_network(count, seed=int(seed) * 100003 + 1 + int(cycle))
        raise ValueError(f"unknown network kind {kind!r}: lattice, random-fixed, random-moving")

    def density(self, stride=None):
        """Observed fraction of the interior."""
        return float(self.lattice(stride).size) / self.grid.n_interior

    # ------------------------------------------------------------------
    def errors(self, x, x_true):
        """RMSE per field over the **interior**, in normalized units.

        The boundary is zero in both fields for every state, so including it
        would only dilute the error by a known factor. ``rmse`` is the error
        in ``q``, the estimated variable; ``psi`` is reported for reference.
        """
        out = {}
        m = self.grid.interior
        for k, b in self.blocks.items():
            d = (x[b][m] - x_true[b][m]) / self.spread[k]
            out[f"rmse_{k}"] = float(np.sqrt(np.mean(d ** 2)))
            out[f"rmse_{k}_raw"] = float(out[f"rmse_{k}"] * self.spread[k])
            ref = np.sqrt(np.mean((x_true[b][m] / self.spread[k]) ** 2))
            out[f"rel_{k}"] = float(out[f"rmse_{k}"] / ref) if ref > 0 else np.nan
        out["rmse"] = out["rmse_q"]
        return out
