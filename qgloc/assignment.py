# -*- coding: utf-8 -*-
"""
Localization as an assignment problem.

Every grid point grows a neighbourhood until it has collected a small number of
candidate observations. It is then updated by **exactly one** of them, or by
none at all. Which one is the decision variable; the radius of the point falls
out of that choice, because the radius is the distance to the observation it
ended up using.

Why this is different from choosing a radius directly
-----------------------------------------------------
A radius per grid point is a free parameter with nothing to pin it down, and
the experiments that preceded this module showed what happens: with one radius
per component the criterion has far more freedom than the data can support, and
optimizing it well makes the analysis worse. Here the parameter is a choice
among a handful of observations that are actually there. There is no radius to
invent, and the search space is the product of a few small sets rather than a
box of integers.

Distance does not decide it
---------------------------
The nearest observation is not automatically the best one. What determines how
much an observation tells you about a point is the ensemble correlation between
them, and in a flow with structure two points aligned along a current can be
far better correlated than two adjacent points sitting across a front. So the
neighbourhood keeps growing past the first observation until it has
``n_candidates`` of them, and the choice among those is made on the cost, not
on the distance.

Abstaining is a legal choice
----------------------------
If none of the candidates correlates with the point, updating it from any of
them injects noise, which is the thing localization exists to prevent. Leaving
the point at its background value is therefore one of the options, and the cost
selects it on its own: with no observation the departure term is zero and only
the residual remains.

The cost
--------
For a point ``i`` updated by a single observation ``j``, the local analysis is a
scalar Kalman update and the cost is the variational one evaluated at it,

    J_i = (x^a_i - x^b_i)^2 / var_i  +  (y_j - x^a_j)^2 / R_j

the departure from the background weighted by the background precision, plus
the residual weighted by the observation precision. No truth, nothing withheld.
Everything is computed from ensemble anomalies, so it costs a handful of dot
products over the N members; the modified-Cholesky precision is never formed
during the search, because the structure of its predecessor sets is what the
assignment is deciding. That structure is built once, at the end, from the
assignment that won.

The global score is the mean of the local costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NONE = -1          # the abstain option, stored in an assignment vector


# ----------------------------------------------------------------------
@dataclass
class Candidates:
    """For each point, the observations it may be updated by.

    ``obs`` has shape (n_points, n_candidates) and holds indices into the
    observation vector, padded with ``NONE`` where a point found fewer than
    asked for. ``dist`` holds the corresponding grid distances, which become
    the radius of the point once one of them is chosen. ``free`` lists the
    points with a real decision to make: a point with a single candidate has
    nothing to optimize and is excluded from the search entirely.
    """
    obs: np.ndarray
    dist: np.ndarray
    free: np.ndarray
    n_candidates: int
    r_max: int

    @property
    def n_points(self):
        return self.obs.shape[0]

    def count(self, i):
        return int(np.sum(self.obs[i] != NONE))

    def summary(self):
        n = np.array([self.count(i) for i in range(self.n_points)])
        return dict(n_points=int(self.n_points),
                    n_free=int(self.free.size),
                    mean_candidates=float(n.mean()),
                    n_with_none=int(np.sum(n == 0)),
                    radius_mean=float(np.mean(self.dist[self.dist > 0])),
                    radius_max=float(np.max(self.dist)))


def build_candidates(g, obs_rows, obs_cols, n_candidates=4, r_max=12,
                     active=None):
    """Grow a square neighbourhood around every point until it has candidates.

    The radius adapts to the local density on its own: where observations are
    dense it stops early, where they are sparse it grows. That is the behaviour
    a fixed radius cannot have, and it costs no extra parameter.

    Distances are Chebyshev, matching the square neighbourhoods the
    predecessor sets use (``geometry.Grid``). Points outside ``active`` --
    the Dirichlet boundary -- get no candidates at all, so they abstain and
    their radius is zero.
    """
    obs_rows = np.asarray(obs_rows, dtype=int)
    obs_cols = np.asarray(obs_cols, dtype=int)
    n_obs = obs_rows.size
    n_points = g * g

    grid = np.full((g, g), NONE, dtype=int)
    grid[obs_rows, obs_cols] = np.arange(n_obs)

    obs = np.full((n_points, n_candidates), NONE, dtype=int)
    dist = np.zeros((n_points, n_candidates), dtype=int)

    active = (np.ones(n_points, dtype=bool) if active is None
              else np.asarray(active, dtype=bool))
    for i in range(n_points):
        if not active[i]:
            continue
        r0, c0 = divmod(i, g)
        found, fdist = [], []
        for r in range(0, r_max + 1):
            if len(found) >= n_candidates:
                break
            lo_r, hi_r = max(0, r0 - r), min(g - 1, r0 + r)
            lo_c, hi_c = max(0, c0 - r), min(g - 1, c0 + r)
            if r == 0:
                ring = [(r0, c0)]
            else:
                ring = []
                for c in range(lo_c, hi_c + 1):
                    if r0 - r >= 0:
                        ring.append((r0 - r, c))
                    if r0 + r <= g - 1:
                        ring.append((r0 + r, c))
                for rr in range(max(lo_r, r0 - r + 1), min(hi_r, r0 + r - 1) + 1):
                    if c0 - r >= 0:
                        ring.append((rr, c0 - r))
                    if c0 + r <= g - 1:
                        ring.append((rr, c0 + r))
            for rr, cc in ring:
                k = grid[rr, cc]
                if k != NONE and k not in found:
                    found.append(int(k))
                    fdist.append(r)
        m = min(len(found), n_candidates)
        obs[i, :m] = found[:m]
        dist[i, :m] = fdist[:m]

    # Only points with more than one candidate carry a decision.
    n_each = np.sum(obs != NONE, axis=1)
    free = np.flatnonzero(n_each > 1)
    return Candidates(obs=obs, dist=dist, free=free,
                      n_candidates=n_candidates, r_max=r_max)


# ----------------------------------------------------------------------
class LocalCost:
    """The variational cost of an assignment, from ensemble anomalies alone.

    Everything is precomputed once per cycle: the variance of each point, the
    variance at each observed site, and the covariance between each point and
    each of its own candidates. Scoring an assignment is then a lookup and a
    mean, and changing one point's choice changes exactly one term. That is
    what makes a local move cheap and the whole search affordable.
    """

    def __init__(self, DX, cand, obs_idx, y, xb, obs_var):
        self.cand = cand
        n_members = DX.shape[1]
        self.var = DX.var(axis=1)                       # background variance
        self.obs_idx = np.asarray(obs_idx, dtype=int)
        self.y = np.asarray(y, dtype=float)
        self.xb = np.asarray(xb, dtype=float)
        self.obs_var = (np.full(self.y.size, float(obs_var))
                        if np.ndim(obs_var) == 0 else np.asarray(obs_var, float))
        self.innov = self.y - self.xb[self.obs_idx]

        A = DX - DX.mean(axis=1, keepdims=True)
        denom = max(n_members - 1, 1)

        # Covariance of each point with each of its candidates, and the
        # variance at the observed sites. Computed once; the search only reads.
        n_p, n_c = cand.obs.shape
        self.cov = np.zeros((n_p, n_c))
        for c in range(n_c):
            k = cand.obs[:, c]
            ok = k != NONE
            if not np.any(ok):
                continue
            site = self.obs_idx[k[ok]]
            self.cov[ok, c] = np.einsum("ij,ij->i", A[np.flatnonzero(ok)],
                                        A[site]) / denom
        self.var_at_obs = self.var[self.obs_idx]

        # The cost of abstaining: the background is kept, so the departure term
        # vanishes and only the residual against the observation remains.
        self._cost_none = np.zeros(n_p)
        for c in range(n_c):
            k = cand.obs[:, c]
            ok = k != NONE
            self._cost_none[ok] = (self.innov[k[ok]] ** 2
                                   / self.obs_var[k[ok]])
        self._table = self._build_table()

    def _build_table(self):
        """Cost of every (point, choice) pair, including abstaining."""
        n_p, n_c = self.cand.obs.shape
        tab = np.full((n_p, n_c + 1), np.inf)
        tab[:, n_c] = self._cost_none
        for c in range(n_c):
            k = self.cand.obs[:, c]
            ok = np.flatnonzero(k != NONE)
            if ok.size == 0:
                continue
            kk = k[ok]
            s = self.var_at_obs[kk] + self.obs_var[kk]
            gain = self.cov[ok, c] / np.where(s > 0, s, 1.0)
            d = gain * self.innov[kk]                    # x^a - x^b
            resid = self.innov[kk] * (1.0 - self.var_at_obs[kk] / s)
            v = np.where(self.var[ok] > 0, self.var[ok], np.inf)
            tab[ok, c] = d ** 2 / v + resid ** 2 / self.obs_var[kk]
        # A point with no candidate at all has only the abstain option, and
        # the cost of that option is defined through its nearest candidate;
        # with none, it is zero.
        none_at_all = np.all(self.cand.obs == NONE, axis=1)
        tab[none_at_all, n_c] = 0.0
        return tab

    # ------------------------------------------------------------------
    @property
    def n_choices(self):
        return self.cand.n_candidates + 1

    def cost_of(self, point, choice):
        return float(self._table[point, choice])

    def score(self, assignment):
        """The global cost: the mean of the local costs."""
        rows = np.arange(self._table.shape[0])
        return float(np.mean(self._table[rows, assignment]))

    def best_greedy(self):
        """The optimal assignment.

        The global cost is a mean of terms each depending only on its own
        point's choice, so the minimum is reached point by point: each takes
        the cheapest of its ``kappa + 1`` options. Nothing has to be searched.
        """
        return np.argmin(np.where(np.isfinite(self._table), self._table, np.inf),
                         axis=1)

    def radius_field(self, assignment):
        """The radius each point ends up with, given the assignment.

        The radius is not chosen; it is the distance to whichever observation
        the point was assigned to. A point that abstains gets zero.
        """
        n_c = self.cand.n_candidates
        r = np.zeros(self.cand.n_points, dtype=int)
        for c in range(n_c):
            sel = assignment == c
            r[sel] = self.cand.dist[sel, c]
        return r

    def assigned_obs(self, assignment):
        """The observation index updating each point, or NONE where it abstains."""
        n_c = self.cand.n_candidates
        out = np.full(self.cand.n_points, NONE, dtype=int)
        for c in range(n_c):
            sel = assignment == c
            out[sel] = self.cand.obs[sel, c]
        return out


# ----------------------------------------------------------------------
# Assignment rules
#
# Four ways of deciding which observation updates each point. All return the
# assignment vector (one entry per point, a candidate index or NONE), and the
# radius field is read off it with ``LocalCost.radius_field``.
#
#   greedy   the optimum of the cost, exact because the cost is separable
#   nearest  every point takes its nearest observation, the classical choice
#   random   a candidate drawn at random per point, the floor of the comparison
#
# The uniform radius, which is the standard localization the method replaces,
# is not an assignment: every point is updated by every observation within a
# fixed distance. It enters the comparison as a radius field only.
# ----------------------------------------------------------------------
def assign_greedy(cost, **_):
    """The optimal assignment, point by point."""
    return cost.best_greedy()


def assign_nearest(cost, **_):
    """Every point takes its nearest observation (candidate 0).

    A point that found no observation within ``r_max`` -- or that is not
    estimated at all -- abstains, as it must under every rule.
    """
    n_c = cost.cand.n_candidates
    return np.where(cost.cand.obs[:, 0] != NONE, 0, n_c).astype(int)


def assign_random(cost, rng=None, allow_abstain=False, **_):
    """A candidate drawn uniformly at random for every point.

    Points whose neighbourhood found fewer than ``kappa`` observations draw
    only among the candidates they actually have.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_c = cost.cand.n_candidates
    have = np.isfinite(cost._table[:, :n_c])
    a = np.empty(cost.cand.n_points, dtype=int)
    for i in range(a.size):
        opts = np.flatnonzero(have[i])
        if allow_abstain:
            opts = np.append(opts, n_c)
        a[i] = rng.choice(opts) if opts.size else n_c
    return a


def uniform_radius(cost, r):
    """The radius field of the standard scheme: the same ``r`` everywhere."""
    return np.full(cost.cand.n_points, int(r), dtype=int)


def assign_variance(cost, **_):
    """The observation that most reduces the analysis error variance of the point.

    ``V_i(j) = s_ii - s_ij^2 / (s_jj + r_j)``; the point takes the candidate
    with the largest reduction. No innovation enters: the choice is made from
    anomalies alone, like a localization, and the data only enter the update.
    Abstains only when no candidate carries any covariance with the point.
    """
    n_c = cost.cand.n_candidates
    red = np.full((cost.cand.n_points, n_c), -np.inf)
    for c in range(n_c):
        k = cost.cand.obs[:, c]; ok = k != NONE
        red[ok, c] = cost.cov[ok, c] ** 2 / (cost.var_at_obs[k[ok]] + cost.obs_var[k[ok]])
    best = red.argmax(axis=1)
    none = ~np.isfinite(red.max(axis=1)) | (red.max(axis=1) <= 0)
    return np.where(none, n_c, best).astype(int)


RULES = dict(greedy=assign_greedy, nearest=assign_nearest, random=assign_random,
             variance=assign_variance)


def evaluate(cost, rule, **kw):
    """Assignment, its cost and its radius field under one rule."""
    a = RULES[rule](cost, **kw)
    return dict(rule=rule, assignment=a, J=cost.score(a),
                radius=cost.radius_field(a))


# ----------------------------------------------------------------------
def assign_cycle(bed, DXq, xbq, obs_idx, y, rule, rng=None):
    """Everything the assignment produces for one forecast, in one call.

    ``DXq``: anomalies of ``q`` (n, N) in normalized units. ``xbq``: the
    background mean, same units. ``obs_idx``: observed q-indices; ``y``: the
    observations. Returns the candidates, the cost, the assignment under
    ``rule`` and the radius field, plus the per-observation bookkeeping the
    figures use (which observations update nothing, and the ensemble variance
    at every observation site).
    """
    cfg = bed.cfg
    g = bed.g
    grid = bed.grid
    cand = build_candidates(g, grid.rows[obs_idx], grid.cols[obs_idx],
                            n_candidates=cfg.kappa, r_max=cfg.r_max,
                            active=grid.interior)
    cost = LocalCost(DXq, cand, obs_idx, y, xbq,
                     np.full(obs_idx.size, cfg.obs_std ** 2))
    a = RULES[rule](cost, rng=rng)
    assigned = cost.assigned_obs(a)
    used = np.zeros(obs_idx.size, dtype=bool)
    used[assigned[assigned != NONE]] = True
    return dict(cand=cand, cost=cost, assignment=a, J=cost.score(a),
                radius=cost.radius_field(a), assigned_obs=assigned,
                obs_used=used, obs_var_ens=cost.var_at_obs.copy(),
                innovation=cost.innov.copy(), table=cost._table.copy())


# ----------------------------------------------------------------------
def assign_partial(bed, DXq, obs_idx, rho=None, r_cap=None, floor=1):
    """The method: radii read from the partial correlations of a wide-radius
    modified-Cholesky precision.

    A precision ``P = L'DL`` is built with the same radius ``rho`` for every
    component (the *probe*; it is never inverted and never used to analyse).
    Its entries normalised, ``-P_ij / sqrt(P_ii P_jj)``, are the partial
    correlations: how much ``i`` depends on the site ``j`` given everything
    else. Every interior component takes the observed site with the largest
    absolute partial correlation -- an observed component takes its own site,
    which the measurements say is right -- and its radius is the Chebyshev
    distance to that site, at least ``floor``. ``P`` has entries up to
    distance ``2 rho`` (two predecessors of one component are conditionally
    linked), so radii above ``rho`` can occur; ``r_cap`` bounds them.

    Nothing about the observation *values* enters: this is a localization.
    The probe radius ``rho`` is the one parameter and it is tied to ``N``:
    each regression has about ``rho^2`` predecessors and must stay determined.
    """
    from .precision import PrecisionBuilder
    cfg, grid = bed.cfg, bed.grid
    rho = int(cfg.rho if rho is None else rho)
    pb = PrecisionBuilder(grid, alpha=cfg.ridge_alpha).bind(DXq)
    P = pb.build(np.where(grid.interior, rho, 0)).tocsr()
    d = np.sqrt(np.maximum(P.diagonal(), 1e-300))
    rho_ij = -P[:, obs_idx].toarray() / (d[:, None] * d[obs_idx][None, :])
    rho_ij[~grid.interior, :] = 0.0
    strength = np.abs(rho_ij)
    best = strength.argmax(axis=1)
    has = strength.max(axis=1) > 0
    assigned = np.where(has, best, NONE)
    dist = np.zeros(grid.n, dtype=int)
    for i in np.flatnonzero(has):
        dist[i] = grid.chebyshev(i, obs_idx[assigned[i]])
    rf = np.where(has, np.maximum(dist, int(floor)), 0)
    if r_cap is not None:
        rf = np.minimum(rf, int(r_cap))
    rf = np.where(grid.interior, rf, 0)
    # bookkeeping shared with the candidate-based rules
    used = np.zeros(obs_idx.size, dtype=bool); used[assigned[assigned != NONE]] = True
    # nearest observed site of every point, for the not-nearest statistic
    rr, cc = grid.rows[obs_idx], grid.cols[obs_idx]
    nearest = np.full(grid.n, NONE, dtype=int)
    inter = np.flatnonzero(grid.interior)
    for i in inter:
        nearest[i] = int(np.argmin(np.maximum(np.abs(rr - grid.rows[i]), np.abs(cc - grid.cols[i]))))
    free = inter
    return dict(cand=type("Cand", (), dict(free=free, n_candidates=obs_idx.size))(),
                cost=None, assignment=(assigned != nearest).astype(int), J=float("nan"),
                radius=rf, assigned_obs=assigned, obs_used=used,
                obs_var_ens=DXq.var(axis=1)[obs_idx], innovation=np.zeros(obs_idx.size),
                table=np.zeros((0, 0)), strength=strength.max(axis=1).astype(np.float32),
                probe_nnz=int(P.nnz), rho=rho)
