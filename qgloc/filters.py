# -*- coding: utf-8 -*-
"""
The two filters, each taking one radius per component, and the cycle.

Both analyses act on the normalized ``q`` block only: ``psi`` is a diagnostic
the model recomputes at every step, so an analysis of it would be thrown away
(see ``testbed.py``). After the analysis ``psi`` is rebuilt from the analysed
``q`` for every member, once, before the next propagation.

**EnKF-MC.** The precision ``B^{-1}`` from ``precision.py`` with the radius
field defining each component's predecessors, then the perturbed-observation
update in precision form,

    (B^{-1} + H' R^{-1} H) Z = H' R^{-1} (Y + E - H X)

solved sparse. The radius field *is* the structure of ``B^{-1}``; that is
what the assignment is designing.

**LETKF.** The standard local ensemble transform, one local analysis per
interior component over its square neighbourhood of radius ``r_i``. Same
neighbourhoods as the predecessor sets, not wrapped at the edges, boundary
excluded. Written here rather than taken from pyteda because pyteda's
neighbourhoods are periodic and its LETKF acts on the full state.

The two above are the references: localization by a fixed radius, every
observation correcting every component within reach. The method is the
three below, where each component is updated by the one observation the
assignment gave it, or by none.

**EnKF-MC-masked.** ``B = (L'DL)^{-1}`` from the global modified Cholesky,
with the assignment radii defining the predecessors. Component ``i`` is
updated with the scalar Kalman gain of its observation under that ``B``:
``dx_i = B_{i,a(i)} d_{a(i)} / (B_{a(i)a(i)} + R)``. One sparse solve per
used observation gives the columns of ``B``; orphan observations are never
solved for. This is the cost ``J`` of the assignment executed as an
analysis, with the Cholesky covariance in place of the raw ensemble one.

**EnKF-MC-group.** The assignment partitions the interior into one group per
observation. Each group is analysed with its observation alone, on a domain
that is the group plus a halo of ``h`` neighbours, with a modified-Cholesky
precision estimated on that domain; the increments of the halo are
discarded. Groups are disjoint, so the analyses are independent and their
order is irrelevant. The halo exists because the Cholesky estimates ``B`` by
regression on neighbours, and a component on the edge of a group would
otherwise lose predecessors.

**LETKF-only.** The local domain of component ``i`` is ``{a(i)}``: the
scalar update with the raw ensemble covariance. No halo is needed, because
nothing is estimated from neighbours.

Masked and LETKF-only differ in one thing only: which covariance the scalar
gain is taken from.

Divergence is recorded, not avoided: a sparse network with a short radius can
give the model gradients it cannot integrate, and Sakov and Oke leave those
cells blank in their figures for the same reason.

Divergence is recorded, not avoided: a sparse network with a short radius can
give the model gradients it cannot integrate, and Sakov and Oke leave those
cells blank in their figures for the same reason.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from .assignment import NONE, assign_cycle, assign_partial
from .flow import FlowGrid
from .lagged import LaggedGrid
from .lasso_struct import LassoGrid
from .clim_struct import ClimGrid
from .bayes_rows import BayesRows
from .shrink_rows import ShrinkRows
from .precision import PrecisionBuilder

FIXED = "fixed"        # arm ("fixed", r): the uniform-radius baseline
PARTIAL = "partial"    # arm ("partial", rho): the method
RULES = ("greedy", "nearest", "random", "variance")   # candidate-based rules, kept for diagnosis
BASELINES = ("enkf-mc", "letkf")                      # take a fixed radius
FLOW = "flow"          # arm ("flow", None): the precision structure from the flow (enkf-mc-flow)
METHODS = ("enkf-mc-masked", "enkf-mc-group", "letkf-only")   # take a rule


class Diverged(RuntimeError):
    """The forecast could not be propagated."""


# ----------------------------------------------------------------------
def forecast(bed, X, x_true):
    """Propagate the ensemble and the truth over one cycle, inflate."""
    cfg = bed.cfg
    T = np.array([0.0, cfg.obs_freq])
    N = X.shape[1]
    Xf = np.stack([bed.model.propagate(X[:, e], T) for e in range(N)], axis=1)
    if not np.all(np.isfinite(Xf)):
        raise Diverged("non-finite forecast")
    xb = Xf.mean(axis=1)
    Xf = xb[:, None] + cfg.inflation * (Xf - xb[:, None])
    x_true = bed.model.propagate(x_true, T)
    return Xf, x_true


def observe(bed, x_true, obs_idx, rng, H=None):
    """Observations at ``obs_idx`` with the configured noise: of normalized q, or of
    normalized psi through the dense operator ``H`` (5% of psi's climatological spread)."""
    xtn = bed.normalize(x_true)[bed.qblock]
    clean = xtn[obs_idx] if H is None else H @ xtn
    return clean + bed.cfg.obs_std * rng.standard_normal(obs_idx.size)


def _analysis_solve(cfg, Qn, Binv, obs_idx, y, rng, H=None):
    """Perturbed-observation analysis in precision form; sparse for point obs, dense for psi."""
    nq, N = Qn.shape; p = obs_idx.size; rinv = 1.0 / cfg.obs_std ** 2
    E = cfg.obs_std * rng.standard_normal((p, N))
    if H is None:
        Hs = sps.csr_matrix((np.ones(p), (np.arange(p), obs_idx)), shape=(p, nq))
        A = (Binv + rinv * (Hs.T @ Hs)).tocsc()
        D = (y[:, None] + E) - (Hs @ Qn)
        Z = spsolve(A, sps.csc_matrix(rinv * (Hs.T @ D)))
        Z = Z.toarray() if sps.issparse(Z) else np.asarray(Z)
    else:
        A = Binv.toarray() + rinv * (H.T @ H)
        D = (y[:, None] + E) - (H @ Qn)
        Z = np.linalg.solve(A, rinv * (H.T @ D))
    return Qn + Z.reshape(nq, N)


# ----------------------------------------------------------------------
def analysis_enkf_mc(bed, Qn, r_field, obs_idx, y, rng, pb=None, H=None):
    """Perturbed-observation EnKF in precision form. ``Qn``: (nq, N) normalized q.
    ``H`` dense (p x nq) for psi observations, None for point observations of q."""
    cfg = bed.cfg
    DX = Qn - Qn.mean(axis=1, keepdims=True)
    if pb is None:
        pb = PrecisionBuilder(bed.grid, alpha=cfg.ridge_alpha)
    pb.bind(DX)
    Binv = pb.build(r_field)
    return _analysis_solve(cfg, Qn, Binv, obs_idx, y, rng, H=H), Binv


def analysis_letkf(bed, Qn, r_field, obs_idx, y, assigned=None, H=None):
    """LETKF over square interior neighbourhoods. ``Qn``: (nq, N) normalized q.

    With ``assigned`` (observation index per component, or NONE) the local
    domain of every component is its assigned observation alone: LETKF-only.
    """
    cfg = bed.cfg
    grid = bed.grid
    nq, N = Qn.shape
    qb = Qn.mean(axis=1)
    DX = Qn - qb[:, None]
    r = np.floor(np.asarray(r_field)).astype(int)
    rinv = 1.0 / cfg.obs_std ** 2

    # observation lookup: state index -> observation row, -1 if unobserved
    where = np.full(nq, -1, dtype=int)
    where[obs_idx] = np.arange(obs_idx.size)
    if H is None:
        innov = y - qb[obs_idx]; Yall = DX[obs_idx, :]
    else:
        innov = y - H @ qb; Yall = H @ DX          # psi anomalies at the sites

    Qa = Qn.copy()
    for i in np.flatnonzero(grid.interior):
        if assigned is not None:
            if assigned[i] == NONE:
                continue
            o = np.array([int(assigned[i])])
        else:
            nb = grid.neighbours(i, int(r[i]), interior_only=True)
            o = where[nb]
            o = o[o >= 0]
        if o.size == 0:
            continue
        Yb = Yall[o, :]                              # (m, N)
        C = Yb.T * rinv                              # (N, m)
        Pa_inv = (N - 1) * np.eye(N) + C @ Yb        # (N, N)
        w, V = np.linalg.eigh(Pa_inv)
        Pa = (V / w) @ V.T
        wa = Pa @ (C @ innov[o])
        Wa = (V / np.sqrt(w)) @ V.T * np.sqrt(N - 1)
        Qa[i, :] = qb[i] + DX[i, :] @ (wa[:, None] + Wa)
    return Qa



def analysis_enkf_mc_masked(bed, Qn, r_field, obs_idx, y, assigned, rng, pb=None,
                            keep_columns=()):
    """Each component updated by its assigned observation under the Cholesky B."""
    from scipy.sparse.linalg import splu
    cfg = bed.cfg
    nq, N = Qn.shape
    DX = Qn - Qn.mean(axis=1, keepdims=True)
    if pb is None:
        pb = PrecisionBuilder(bed.grid, alpha=cfg.ridge_alpha)
    pb.bind(DX)
    Binv = pb.build(r_field)
    used = np.unique(assigned[assigned != NONE])
    Qa = Qn.copy()
    cols = {}
    if used.size:
        lu = splu(Binv.tocsc())
        E = np.zeros((nq, used.size)); E[obs_idx[used], np.arange(used.size)] = 1.0
        Bc = lu.solve(E)                                  # columns of B at used obs
        R = cfg.obs_std ** 2
        noise = cfg.obs_std * rng.standard_normal((obs_idx.size, N))
        pos = {int(j): k for k, j in enumerate(used)}
        for i in np.flatnonzero(assigned != NONE):
            j = int(assigned[i]); k = pos[j]
            g = Bc[i, k] / (Bc[obs_idx[j], k] + R)
            Qa[i, :] = Qn[i, :] + g * ((y[j] + noise[j, :]) - Qn[obs_idx[j], :])
        for j in keep_columns:
            if int(j) in pos:
                cols[int(j)] = Bc[:, pos[int(j)]].astype(np.float32)
    return Qa, Binv, cols


def group_domains(bed, assigned, obs_idx, halo):
    """The domain of every observation: its group's window plus ``halo``."""
    grid = bed.grid
    out = {}
    for j in np.unique(assigned[assigned != NONE]):
        members = np.flatnonzero(assigned == j)
        c = int(obs_idx[j])
        ext = max(grid.chebyshev(i, c) for i in members)
        out[int(j)] = (members, grid.neighbours(c, ext + int(halo), interior_only=True))
    return out


def analysis_enkf_mc_group(bed, Qn, r_field, obs_idx, y, assigned, rng, halo=2):
    """Each group analysed with its observation on group + halo; halo discarded."""
    from .precision import ridge_scaled
    cfg = bed.cfg
    grid = bed.grid
    nq, N = Qn.shape
    DX = Qn - Qn.mean(axis=1, keepdims=True)
    r = np.floor(np.asarray(r_field)).astype(int)
    R = cfg.obs_std ** 2; rinv = 1.0 / R
    noise = cfg.obs_std * rng.standard_normal((obs_idx.size, N))
    Qa = Qn.copy()
    domains = group_domains(bed, assigned, obs_idx, halo)
    for j, (members, D) in domains.items():
        m = D.size
        loc = {int(g): k for k, g in enumerate(D)}
        L = np.eye(m); d = np.empty(m)
        for k, i in enumerate(D):
            yk = DX[i, :]
            pred = grid.predecessors(int(i), int(r[i])) if r[i] > 0 else np.empty(0, dtype=int)
            pred = np.array([q for q in pred if int(q) in loc], dtype=int)
            v = float(np.var(yk))
            if pred.size and v > 0:
                X = DX[pred, :].T
                beta = ridge_scaled(X, yk, cfg.ridge_alpha)
                L[k, [loc[int(q)] for q in pred]] = -beta
                v = float(np.var(yk - X @ beta))
            d[k] = 1.0 / v if v > 0 else 0.0
        Binv = L.T @ (d[:, None] * L)
        kj = loc[int(obs_idx[j])]
        A = Binv.copy(); A[kj, kj] += rinv
        rhs = np.zeros((m, N)); rhs[kj, :] = rinv * ((y[j] + noise[j, :]) - Qn[obs_idx[j], :])
        Z = np.linalg.solve(A, rhs)
        ks = [loc[int(i)] for i in members]
        Qa[members, :] = Qn[members, :] + Z[ks, :]
    return Qa, domains

# ----------------------------------------------------------------------
def radius_field(bed, arm, DXq, xbq, obs_idx, y, rng):
    """The radius field of one arm for one forecast, plus assignment output.

    ``arm`` is ``("fixed", r)``, ``("partial", rho)`` or ``(<rule>, None)``.
    """
    kind, r = arm
    if kind == FIXED:
        rf = np.where(bed.grid.interior, int(r), 0)
        return rf, None
    if kind in (FLOW, "lagged", "lasso", "clim", "bayes", "shrink", "climstart"):
        return np.where(bed.grid.interior, 1, 0), None      # the structure is built in the analysis
    if kind == PARTIAL:
        out = assign_partial(bed, DXq, obs_idx, rho=r,
                             r_cap=getattr(bed.cfg, "r_cap", None))
        return out["radius"], out
    out = assign_cycle(bed, DXq, xbq, obs_idx, y, rule=kind, rng=rng)
    return out["radius"], out


def run_cycles(bed, X0, x_true0, filt, arm, seed, network=None, recorder=None,
               cycles=None):
    """A filter run. ``filt`` in {"enkf-mc", "letkf"}. Returns per-cycle rows."""
    cfg = bed.cfg
    rng = np.random.default_rng(seed)
    rng_rule = np.random.default_rng(seed + 7919)
    qb = bed.qblock
    X, x_true = X0.copy(), x_true0.copy()
    pb = PrecisionBuilder(bed.grid, alpha=cfg.ridge_alpha) if filt.startswith("enkf-mc") else None
    halo = int(getattr(cfg, "halo", 2))
    snap = set(getattr(recorder, "snap_cycles", ()))
    rows = []
    n_cyc = int(cycles or cfg.cycles)
    DXa_prev = None
    clim_grid = None
    bayes = None
    shrink = None
    spread0 = None
    for k in range(n_cyc):
        try:
            Xf, x_true = forecast(bed, X, x_true)
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            rows.append(dict(cycle=k, diverged=True))
            if recorder is not None:
                recorder.diverged(k)
            break
        obs_idx = bed.network(cycle=k, seed=seed, kind=network)
        Hobs = bed.obs_operator(obs_idx)
        y = observe(bed, x_true, obs_idx, rng, H=Hobs)

        Qn = bed.normalize(Xf)[qb, :]
        xbq = Qn.mean(axis=1)
        DXq = Qn - xbq[:, None]
        t0 = time.time()
        rf, aout = radius_field(bed, arm, DXq, xbq, obs_idx, y, rng_rule)
        t_assign = time.time() - t0

        t0 = time.time()
        Binv = None; extra = {}; rec_flow = {}
        if filt == "enkf-mc":
            Qa, Binv = analysis_enkf_mc(bed, Qn, rf, obs_idx, y, rng, pb=pb, H=Hobs)
        elif filt == "letkf":
            Qa = analysis_letkf(bed, Qn, rf, obs_idx, y, H=Hobs)
        elif filt == "enkf-mc-flow":
            psi = bed.psi_from_q(Xf.mean(axis=1))[bed.blocks["psi"]]
            fgrid = FlowGrid(bed.grid, psi, cfg.obs_freq, cap=cfg.wake_cap, width=cfg.wake_width, local=cfg.wake_local)
            pbf = PrecisionBuilder(fgrid, alpha=cfg.ridge_alpha)
            Qa, Binv = analysis_enkf_mc(bed, Qn, rf, obs_idx, y, rng, pb=pbf)
            npred = fgrid.n_predecessors()
            extra["n_pred"] = npred.astype(np.int16)
            extra["depth"] = fgrid.depth.astype(np.int16)
            rec_flow = dict(pred_mean=float(npred[bed.grid.interior].mean()), pred_max=int(npred.max()),
                            wake_mean=float(fgrid.wake_length()[bed.grid.interior].mean()))
        elif filt == "enkf-mc-lagged":
            if DXa_prev is None:
                Qa, Binv = analysis_enkf_mc(bed, Qn, np.where(bed.grid.interior, 2, 0), obs_idx, y, rng, pb=pb)
            else:
                lg = LaggedGrid(bed.grid, DXa_prev, DXq, window=cfg.lag_window, c=cfg.lag_c, local=cfg.wake_local)
                Qa, Binv = analysis_enkf_mc(bed, Qn, rf, obs_idx, y, rng, pb=PrecisionBuilder(lg, alpha=cfg.ridge_alpha))
                sup = lg.support
                rec_flow = dict(pred_mean=float(sup[bed.grid.interior].mean()), pred_max=int(sup.max()))
        elif filt == "enkf-mc-lasso":
            lg = LassoGrid(bed.grid, DXq, window=cfg.lasso_window, c=cfg.lasso_c, local=cfg.wake_local)
            Qa, Binv = analysis_enkf_mc(bed, Qn, rf, obs_idx, y, rng, pb=PrecisionBuilder(lg, alpha=cfg.ridge_alpha))
            sup = lg.support
            rec_flow = dict(pred_mean=float(sup[bed.grid.interior].mean()), pred_max=int(sup.max()))
        elif filt == "enkf-mc-clim":
            if clim_grid is None:
                clim_q = bed.normalize(bed.snapshots.T.astype(float))[bed.qblock, :]
                clim_grid = ClimGrid(bed.grid, clim_q, window=cfg.lasso_window, c=cfg.lasso_c, local=cfg.wake_local)
                pb = PrecisionBuilder(clim_grid, alpha=cfg.ridge_alpha)
                sup = clim_grid.support
                rec_flow = dict(pred_mean=float(sup[bed.grid.interior].mean()), pred_max=int(sup.max()))
            Qa, Binv = analysis_enkf_mc(bed, Qn, rf, obs_idx, y, rng, pb=pb)
        elif filt == "enkf-mc-climstart":
            # the method that survived the weekend: climate structure and climate
            # prior on the coefficients at the first analysis (alpha0), plain
            # uniform radius with ridge toward zero from the second cycle on
            if k < int(cfg.climstart_cycles):
                if shrink is None:
                    clim_q = bed.normalize(bed.snapshots.T.astype(float))[bed.qblock, :]
                    A_c = clim_q - clim_q.mean(axis=1, keepdims=True)
                    cg = ClimGrid(bed.grid, clim_q, window=cfg.lasso_window, c=cfg.lasso_c, local=1)
                    shrink = ShrinkRows(bed.grid, cg, A_c)
                Binv = shrink.build(DXq, alpha=cfg.shrink_alpha0)
                Qa = _analysis_solve(cfg, Qn, Binv, obs_idx, y, rng, H=Hobs)
            else:
                Qa, Binv = analysis_enkf_mc(bed, Qn, np.where(bed.grid.interior, int(cfg.climstart_radius), 0), obs_idx, y, rng, pb=pb)
        elif filt == "enkf-mc-shrink":
            if shrink is None:
                clim_q = bed.normalize(bed.snapshots.T.astype(float))[bed.qblock, :]
                A_c = clim_q - clim_q.mean(axis=1, keepdims=True)
                cg = ClimGrid(bed.grid, clim_q, window=cfg.lasso_window, c=cfg.lasso_c, local=1)
                shrink = ShrinkRows(bed.grid, cg, A_c)
            s_now = float(np.mean(np.std(DXq[bed.grid.interior], axis=1)))
            if spread0 is None:
                spread0 = s_now
            if cfg.shrink_alpha0 <= 0:
                al = None
            elif cfg.shrink_decay > 0:
                al = max(cfg.shrink_floor, cfg.shrink_alpha0 * cfg.shrink_decay ** k)
            else:
                al = max(cfg.shrink_floor, cfg.shrink_alpha0 * (s_now / spread0) ** 2)
            Binv = shrink.build(DXq, alpha=al)
            Qa = _analysis_solve(cfg, Qn, Binv, obs_idx, y, rng, H=Hobs)
            la = shrink.last_alpha[bed.grid.interior]
            rec_flow = dict(pred_mean=float(np.nanmedian(la)), pred_max=int(np.nanmax(la)))   # median / max chosen alpha
        elif filt in ("enkf-mc-bayes", "enkf-mc-bayes-uniform"):
            if bayes is None:
                clim_q = bed.normalize(bed.snapshots.T.astype(float))[bed.qblock, :]
                A_c = clim_q - clim_q.mean(axis=1, keepdims=True)
                if filt == "enkf-mc-bayes":
                    cg = ClimGrid(bed.grid, clim_q, window=cfg.lasso_window, c=min(cfg.lasso_c, 1e6), local=cfg.wake_local)
                    struct = lambda i: cg.pred[i]
                else:
                    struct = lambda i: bed.grid.predecessors(i, int(cfg.bayes_radius))
                cw = cfg.bayes_clim_weight if cfg.bayes_clim_weight > 0 else A_c.shape[1] / DXq.shape[1]   # M / N
                bayes = BayesRows(bed.grid, struct, A_c, alpha=cfg.ridge_alpha, rho=cfg.bayes_rho, clim_weight=cw, clim_decay=cfg.bayes_clim_decay, taper=cfg.bayes_taper)
            Binv = bayes.update_and_build(DXq)
            if cfg.bayes_rho < 0:
                lr = bayes.last_rho[bed.grid.interior]
                rec_flow = dict(pred_mean=float(np.nanmean(lr)), pred_max=int(np.nanmax(lr) * 100),
                                frac_low=float(np.mean(lr <= 0.4)), frac_high=float(np.mean(lr >= 0.95)),
                                wc_mean=float(np.mean(bayes.last_wc[bed.grid.interior])), taper=float(bayes.taper))
            Qa = _analysis_solve(cfg, Qn, Binv, obs_idx, y, rng, H=Hobs)
        elif Hobs is not None and filt in ("enkf-mc-masked", "enkf-mc-group", "letkf-only", "enkf-mc-lagged", "enkf-mc-lasso", "enkf-mc-clim"):
            raise ValueError(f"{filt} supports point observations of q only (obs_var='q')")
        elif filt == "enkf-mc-masked":
            keep = aout["obs_used"].nonzero()[0][::max(1, aout["obs_used"].sum() // 3)][:3] if k in snap else ()
            Qa, Binv, cols = analysis_enkf_mc_masked(bed, Qn, rf, obs_idx, y, aout["assigned_obs"], rng,
                                                     pb=pb, keep_columns=keep)
            extra["b_columns"] = cols
        elif filt == "enkf-mc-group":
            Qa, domains = analysis_enkf_mc_group(bed, Qn, rf, obs_idx, y, aout["assigned_obs"], rng, halo=halo)
            extra["domains"] = domains
        elif filt == "letkf-only":
            Qa = analysis_letkf(bed, Qn, rf, obs_idx, y, assigned=aout["assigned_obs"])
        else:
            raise ValueError(filt)
        t_analysis = time.time() - t0
        if not np.all(np.isfinite(Qa)):
            rows.append(dict(cycle=k, diverged=True))
            if recorder is not None:
                recorder.diverged(k)
            break

        Xa = bed.normalize(Xf).copy()
        Xa[qb, :] = Qa
        Xa = bed.denormalize(Xa)
        for e in range(Xa.shape[1]):
            Xa[:, e] = bed.psi_from_q(Xa[:, e])

        xb = Xf.mean(axis=1)
        xa = Xa.mean(axis=1)
        DXa_prev = Qa - Qa.mean(axis=1, keepdims=True)
        rec = dict(cycle=k, diverged=False, p_obs=int(obs_idx.size),
                   spread=float(np.mean(np.std(Qa[bed.grid.interior], axis=1))),
                   t_assign=t_assign, t_analysis=t_analysis,
                   r_mean=float(rf[bed.grid.interior].mean()),
                   r_max=int(rf.max()))
        rec.update(rec_flow)
        rec.update({f"b_{a}": b for a, b in bed.errors(xb, x_true).items()})
        rec.update(bed.errors(xa, x_true))
        if aout is not None:
            rec.update(J=aout["J"],
                       frac_not_nearest=float(np.mean(
                           aout["assignment"][aout["cand"].free] != 0)),
                       frac_abstain=float(np.mean(
                           aout["assigned_obs"][bed.grid.interior] == NONE)),
                       obs_unused=int((~aout["obs_used"]).sum()))
        rows.append(rec)
        if recorder is not None:
            recorder.cycle(k, rec, xb=xb, xa=xa, x_true=x_true, Xa=Xa,
                           obs_idx=obs_idx, y=y, rf=rf, aout=aout, Binv=Binv, extra=extra)
        X = Xa
    return rows


def score(rows, burn_in, key="rmse"):
    """Post-burn-in mean of a per-cycle quantity; NaN if the run diverged."""
    if any(r.get("diverged") for r in rows):
        return float("nan")
    v = [r[key] for r in rows[burn_in:] if key in r]
    return float(np.mean(v)) if v else float("nan")
