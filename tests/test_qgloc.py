# -*- coding: utf-8 -*-
"""Fast tests: no model integration, everything on synthetic ensembles."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from qgloc.assignment import (NONE, RULES, LocalCost, build_candidates, evaluate,
                              uniform_radius)
from qgloc.geometry import Grid
from qgloc.precision import PrecisionBuilder, ridge_scaled

G = 21


@pytest.fixture
def grid():
    return Grid(G)


@pytest.fixture
def ensemble():
    rng = np.random.default_rng(0)
    DX = np.stack([gaussian_filter(rng.standard_normal((G, G)), 1.5).ravel() for _ in range(20)], 1)
    m = Grid(G).interior
    DX[~m] = 0.0
    return DX - DX.mean(axis=1, keepdims=True)


@pytest.fixture
def lattice(grid):
    rr, cc = np.meshgrid(np.arange(2, G - 1, 4), np.arange(2, G - 1, 4), indexing="ij")
    return np.sort(rr.ravel() * G + cc.ravel())


# ---------------------------------------------------------------- geometry
def test_interior_count(grid):
    assert grid.n_interior == (G - 2) ** 2


def test_neighbourhood_does_not_wrap(grid):
    i = 1 * G + 1                      # top-left interior corner
    nb = grid.neighbours(i, 1)
    assert set(nb) == {G + 1, G + 2, 2 * G + 1, 2 * G + 2}
    assert all(grid.interior[nb])


def test_predecessors_below(grid):
    i = 5 * G + 5
    p = grid.predecessors(i, 2)
    assert p.size and np.all(p < i) and np.all(grid.interior[p])


# ---------------------------------------------------------------- precision
def test_ridge_scaled_is_scale_invariant():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 5)); y = X @ np.arange(1, 6) + 0.1 * rng.standard_normal(20)
    b1 = ridge_scaled(X, y, 0.3)
    b2 = ridge_scaled(1e4 * X, 1e4 * y, 0.3)
    assert np.allclose(b1, b2)


def test_precision_boundary_rows_are_unit(grid, ensemble):
    pb = PrecisionBuilder(grid, alpha=0.3).bind(ensemble)
    B = pb.build(np.full(grid.n, 2))
    d = B.diagonal()
    assert np.allclose(d[~grid.interior], 1.0)
    row = B[0].toarray().ravel()
    assert row[0] == 1.0 and np.count_nonzero(row) == 1


def test_precision_is_spd_and_sparse(grid, ensemble):
    pb = PrecisionBuilder(grid, alpha=0.3).bind(ensemble)
    B = pb.build(np.full(grid.n, 1)).toarray()
    assert np.allclose(B, B.T)
    assert np.linalg.eigvalsh(B).min() > 0
    assert np.count_nonzero(B) < 0.05 * grid.n ** 2


def test_radius_zero_is_diagonal_row(grid, ensemble):
    pb = PrecisionBuilder(grid, alpha=0.3).bind(ensemble)
    r = np.zeros(grid.n, dtype=int)
    B = pb.build(r)
    assert B.nnz == grid.n


# ---------------------------------------------------------------- assignment
def test_candidates_respect_active(grid, lattice):
    cand = build_candidates(G, grid.rows[lattice], grid.cols[lattice], 4, 8, active=grid.interior)
    assert np.all(cand.obs[~grid.interior] == NONE)
    assert np.all(np.sum(cand.obs[grid.interior] != NONE, axis=1) >= 1)
    assert np.all(np.diff(cand.dist[grid.interior], axis=1) >= 0)


def _cost(grid, ensemble, lattice):
    rng = np.random.default_rng(2)
    cand = build_candidates(G, grid.rows[lattice], grid.cols[lattice], 4, 8, active=grid.interior)
    xb = np.zeros(grid.n)
    y = 0.05 * rng.standard_normal(lattice.size)
    return LocalCost(ensemble, cand, lattice, y, xb, np.full(lattice.size, 0.05 ** 2))


def test_greedy_is_the_optimum(grid, ensemble, lattice):
    cost = _cost(grid, ensemble, lattice)
    a = cost.best_greedy()
    J = cost.score(a)
    rng = np.random.default_rng(3)
    for _ in range(200):                      # any single move cannot improve
        i = rng.integers(grid.n); c = rng.integers(cost.n_choices)
        b = a.copy(); b[i] = c
        assert cost.score(b) >= J - 1e-12


def test_rules_order(grid, ensemble, lattice):
    cost = _cost(grid, ensemble, lattice)
    Jg = evaluate(cost, "greedy")["J"]
    Jn = evaluate(cost, "nearest")["J"]
    Jr = evaluate(cost, "random", rng=np.random.default_rng(0))["J"]
    assert Jg <= Jn and Jg <= Jr and np.isfinite(Jn) and np.isfinite(Jr)


def test_radius_field_matches_choice(grid, ensemble, lattice):
    cost = _cost(grid, ensemble, lattice)
    out = evaluate(cost, "greedy")
    rf, a = out["radius"], out["assignment"]
    for i in np.flatnonzero(grid.interior)[:50]:
        if a[i] < cost.cand.n_candidates:
            assert rf[i] == cost.cand.dist[i, a[i]]
        else:
            assert rf[i] == 0
    assert np.all(rf[~grid.interior] == 0)
    assert uniform_radius(cost, 3)[0] == 3


# ---------------------------------------------------------------- filters (no model)
def test_enkf_mc_and_letkf_reduce_error(grid, ensemble, lattice):
    """A one-step analysis on a synthetic truth must beat the background."""
    from qgloc.filters import analysis_enkf_mc, analysis_letkf

    class Cfg: obs_std = 0.05; ridge_alpha = 0.3
    class Bed: cfg = Cfg(); pass
    bed = Bed(); bed.grid = grid
    rng = np.random.default_rng(4)
    truth = ensemble[:, 0] * 0.8 + ensemble[:, 1] * 0.6
    Qn = ensemble[:, 2:] + 0.0
    y = truth[lattice] + 0.05 * rng.standard_normal(lattice.size)
    rf = np.where(grid.interior, 2, 0)
    e0 = np.sqrt(np.mean((Qn.mean(1) - truth)[grid.interior] ** 2))
    Qa, _ = analysis_enkf_mc(bed, Qn, rf, lattice, y, rng)
    e1 = np.sqrt(np.mean((Qa.mean(1) - truth)[grid.interior] ** 2))
    Ql = analysis_letkf(bed, Qn, rf, lattice, y)
    e2 = np.sqrt(np.mean((Ql.mean(1) - truth)[grid.interior] ** 2))
    assert e1 < e0 and e2 < e0
    assert np.allclose(Qa[~grid.interior], Qn[~grid.interior])
    assert np.allclose(Ql[~grid.interior], Qn[~grid.interior])


def test_one_observation_methods(grid, ensemble, lattice):
    """masked, group and letkf-only update only assigned components, reduce error."""
    from qgloc.assignment import build_candidates, LocalCost
    from qgloc.filters import (analysis_enkf_mc_masked, analysis_enkf_mc_group,
                               analysis_letkf, group_domains)

    class Cfg: obs_std = 0.05; ridge_alpha = 0.3; kappa = 4; r_max = 8
    class Bed: cfg = Cfg()
    bed = Bed(); bed.grid = grid; bed.g = G
    rng = np.random.default_rng(5)
    truth = ensemble[:, 0] * 0.8 + ensemble[:, 1] * 0.6
    Qn = ensemble[:, 2:] + 0.0
    y = truth[lattice] + 0.05 * rng.standard_normal(lattice.size)
    cand = build_candidates(G, grid.rows[lattice], grid.cols[lattice], 4, 8, active=grid.interior)
    DX = Qn - Qn.mean(1, keepdims=True)
    cost = LocalCost(DX, cand, lattice, y, Qn.mean(1), np.full(lattice.size, 0.05 ** 2))
    # the nearest rule: the J rule is known not to reduce the error (see FINDINGS)
    a = RULES["nearest"](cost); rf = cost.radius_field(a); asg = cost.assigned_obs(a)
    e0 = np.sqrt(np.mean((Qn.mean(1) - truth)[grid.interior] ** 2))
    Qm, _, cols = analysis_enkf_mc_masked(bed, Qn, rf, lattice, y, asg, rng, keep_columns=[int(asg[asg >= 0][0])])
    Qg, dom = analysis_enkf_mc_group(bed, Qn, rf, lattice, y, asg, rng, halo=2)
    Ql = analysis_letkf(bed, Qn, rf, lattice, y, assigned=asg)
    untouched = asg < 0
    for Q in (Qm, Qg, Ql):
        assert np.allclose(Q[untouched], Qn[untouched])
        assert np.sqrt(np.mean((Q.mean(1) - truth)[grid.interior] ** 2)) < e0
    assert len(cols) == 1
    members = np.concatenate([m for m, D in dom.values()])
    assert np.unique(members).size == members.size == int((~untouched).sum())
    for m, D in dom.values():
        assert np.all(np.isin(m, D))


def test_partial_rule(grid, ensemble, lattice):
    """The method: radii from the probe precision; observed points take their own site."""
    from qgloc.assignment import assign_partial

    class Cfg: obs_std = 0.05; ridge_alpha = 0.3; rho = 3; r_cap = None
    class Bed: cfg = Cfg()
    bed = Bed(); bed.grid = grid; bed.g = G
    out = assign_partial(bed, ensemble, lattice, rho=3)
    rf, asg = out["radius"], out["assigned_obs"]
    assert np.all(rf[~grid.interior] == 0)
    assert np.all(rf[grid.interior] >= 1)
    assert rf.max() <= 6                                   # at most 2 * rho
    own = lattice[grid.interior[lattice]]
    assert np.all(asg[own] == np.searchsorted(lattice, own))
    assert np.all(rf[own] == 1)
    capped = assign_partial(bed, ensemble, lattice, rho=3, r_cap=2)["radius"]
    assert capped.max() <= 2
