# -*- coding: utf-8 -*-
"""
EXP-01  The radii on a single forecast, measured without the analysis.

The method (partial-correlation radii, one probe radius per ``scale.rhos``)
and the candidate-based rules of the first draft (J, nearest, random,
variance), side by side, before any analysis.

Everything here is computed from ensemble anomalies and observations, before
any analysis is performed, so it does not depend on which filter follows. For
every seed, both networks, at the first cycle of the benchmark and again after
``burn_in`` cycles of the greedy EnKF-MC (so that the ensemble is one the
filter actually produces, not a climatological draw):

- the cost J of the four rules and of abstaining everywhere
- the fraction of free components that do not take their nearest observation
- the radius field of each rule, and the uniform field of the same mean
- the tessellation: which observation updates each component
- which observations update nothing, and the ensemble variance at every site
- the sparsity of B^-1 for each radius field and for the uniform one
- the cost of evaluating J against the cost of assembling B^-1

Written to ``results/EXP-01_<scale>/<network>/<s<stride>|d<pct>>/seed<seed>/cycle<k>.npz`` and
summarised in ``single_cycle.csv``.
"""
from __future__ import annotations

import os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ExperimentContext, get_scale, make_testbed, parse_cli, shard_cells
from qgloc.assignment import NONE, LocalCost, RULES, assign_partial, build_candidates, uniform_radius
from qgloc.filters import forecast, observe, run_cycles
from qgloc.precision import PrecisionBuilder
from qgloc.progress import Progress, log

EXP_ID = "EXP-01"


def diagnose(bed, Xf, x_true, obs_idx, y, rng, rhos=(4,)):
    cfg, grid = bed.cfg, bed.grid
    qb = bed.qblock
    Qn = bed.normalize(Xf)[qb, :]
    xbq = Qn.mean(axis=1)
    DX = Qn - xbq[:, None]

    t0 = time.time()
    cand = build_candidates(bed.g, grid.rows[obs_idx], grid.cols[obs_idx],
                            n_candidates=cfg.kappa, r_max=cfg.r_max, active=grid.interior)
    cost = LocalCost(DX, cand, obs_idx, y, xbq, np.full(obs_idx.size, cfg.obs_std ** 2))
    t_table = time.time() - t0

    out = dict(obs_idx=obs_idx.astype(np.int32), y=y.astype(np.float32),
               xb_q=Xf.mean(axis=1)[qb].astype(np.float32),
               xt_q=x_true[qb].astype(np.float32),
               ens_var=DX.var(axis=1).astype(np.float32),
               obs_var_ens=cost.var_at_obs.astype(np.float32),
               innovation=cost.innov.astype(np.float32),
               cand_obs=cand.obs.astype(np.int32), cand_dist=cand.dist.astype(np.int16),
               free=cand.free.astype(np.int32), cost_table=cost._table.astype(np.float32))

    # timing of J
    a0 = cost.best_greedy()
    t0 = time.time()
    for _ in range(200):
        cost.score(a0)
    t_J = (time.time() - t0) / 200

    stats = dict(n_obs=int(obs_idx.size), t_table=t_table, t_J=t_J,
                 J_abstain=float(cost.score(np.full(grid.n, cand.n_candidates))))
    pb = PrecisionBuilder(grid, alpha=cfg.ridge_alpha).bind(DX)
    fields = {}
    for rule in RULES:
        a = RULES[rule](cost, rng=rng)
        rf = cost.radius_field(a)
        assigned = cost.assigned_obs(a)
        used = np.zeros(obs_idx.size, dtype=bool)
        used[assigned[assigned != NONE]] = True
        t0 = time.time()
        B = pb.build(rf)
        t_B = time.time() - t0
        fields[rule] = rf
        out[f"assignment_{rule}"] = a.astype(np.int8)
        out[f"radius_{rule}"] = rf.astype(np.int16)
        out[f"assigned_obs_{rule}"] = assigned.astype(np.int32)
        out[f"obs_used_{rule}"] = used
        out[f"binv_{rule}_indptr"] = B.indptr.astype(np.int32)
        out[f"binv_{rule}_indices"] = B.indices.astype(np.int32)
        stats.update({f"J_{rule}": float(cost.score(a)),
                      f"r_mean_{rule}": float(rf[grid.interior].mean()),
                      f"r_max_{rule}": int(rf.max()),
                      f"not_nearest_{rule}": float(np.mean(a[cand.free] != 0)),
                      f"abstain_{rule}": float(np.mean(assigned[grid.interior] == NONE)),
                      f"obs_unused_{rule}": int((~used).sum()),
                      f"nnz_{rule}": int(B.nnz), f"t_B_{rule}": t_B})
    # the method: partial-correlation radii from the probe precision
    for rho in rhos:
        t0 = time.time()
        pa = assign_partial(bed, DX, obs_idx, rho=rho, r_cap=cfg.r_cap)
        t_probe = time.time() - t0
        rf = pa["radius"]; assigned = pa["assigned_obs"]
        t0 = time.time(); B = pb.build(rf); t_B = time.time() - t0
        tag = f"partial{rho}"
        out[f"radius_{tag}"] = rf.astype(np.int16)
        out[f"assigned_obs_{tag}"] = assigned.astype(np.int32)
        out[f"obs_used_{tag}"] = pa["obs_used"]
        out[f"strength_{tag}"] = pa["strength"]
        out[f"binv_{tag}_indptr"] = B.indptr.astype(np.int32)
        out[f"binv_{tag}_indices"] = B.indices.astype(np.int32)
        stats.update({f"r_mean_{tag}": float(rf[grid.interior].mean()),
                      f"r_max_{tag}": int(rf.max()),
                      f"not_nearest_{tag}": float(np.mean(pa["assignment"][grid.interior])),
                      f"obs_unused_{tag}": int((~pa["obs_used"]).sum()),
                      f"nnz_{tag}": int(B.nnz), f"probe_nnz_{tag}": pa["probe_nnz"],
                      f"t_probe_{tag}": t_probe, f"t_B_{tag}": t_B})
    # uniform field with the same mean radius as the method
    r_u = int(round(stats.get(f"r_mean_partial{rhos[0]}", stats["r_mean_greedy"])))
    rf_u = np.where(grid.interior, r_u, 0)
    B = pb.build(rf_u)
    out["radius_uniform"] = rf_u.astype(np.int16)
    out["binv_uniform_indptr"] = B.indptr.astype(np.int32)
    out["binv_uniform_indices"] = B.indices.astype(np.int32)
    stats.update(r_uniform=r_u, nnz_uniform=int(B.nnz))
    # the unused observations: interior distance and ensemble variance
    used = out["obs_used_greedy"]
    v = cost.var_at_obs
    stats.update(var_used_median=float(np.median(v[used])) if used.any() else np.nan,
                 var_unused_median=float(np.median(v[~used])) if (~used).any() else np.nan)
    return out, stats


def main():
    a = parse_cli()
    scale = get_scale(a.scale)
    ctx = ExperimentContext(EXP_ID, "assignment diagnostics on single forecasts", scale)
    cells = []
    for net in scale.networks:
        axis = scale.strides if net == "lattice" else scale.densities
        cells += [(net, v, seed) for v in axis for seed in scale.seeds]
    cells = shard_cells(cells)
    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID)
    for net, v, seed in cells:
        st = f"s{v}" if net == "lattice" else f"d{int(round(100*v))}"
        over = dict(obs_stride=v) if net == "lattice" else dict(obs_density=v)
        bed = make_testbed(scale, obs_network=net, **over)
        cfg = bed.cfg
        rng = np.random.default_rng(seed)
        X0, xt0 = bed.build_ensemble(seed)
        # cycle 0: the climatological draw
        Xf, xt = forecast(bed, X0, xt0)
        obs_idx = bed.network(cycle=0, seed=seed, kind=net)
        y = observe(bed, xt, obs_idx, rng)
        out, st_ = diagnose(bed, Xf, xt, obs_idx, y, rng, rhos=scale.rhos)
        np.savez_compressed(ctx.path(net, st, f"seed{seed}", "cycle0.npz"), **out)
        rows.append(dict(network=net, axis=st, seed=seed, cycle=0, **st_))
        # after burn_in cycles of the greedy EnKF-MC
        k = int(cfg.burn_in)
        state = {}
        def keep(kk, rec, xb, xa, x_true, Xa, **_):
            state["Xa"], state["xt"] = Xa, x_true
        class _R:
            def cycle(self, kk, rec, **kw): keep(kk, rec, **kw)
            def diverged(self, kk): pass
        run_cycles(bed, X0, xt0, "enkf-mc", ("partial", scale.diag_rho), seed=seed, network=net,
                   recorder=_R(), cycles=k)
        if "Xa" in state:
            Xf, xt = forecast(bed, state["Xa"], state["xt"])
            obs_idx = bed.network(cycle=k, seed=seed, kind=net)
            y = observe(bed, xt, obs_idx, rng)
            out, st_ = diagnose(bed, Xf, xt, obs_idx, y, rng, rhos=scale.rhos)
            np.savez_compressed(ctx.path(net, st, f"seed{seed}", f"cycle{k}.npz"), **out)
            rows.append(dict(network=net, axis=st, seed=seed, cycle=k, **st_))
        tag = f"partial{scale.rhos[0]}"
        prog.step(f"{net}/{st}/seed{seed}: {tag} r_mean={st_['r_mean_'+tag]:.2f} r_max={st_['r_max_'+tag]} "
                  f"not-nearest={100*st_['not_nearest_'+tag]:.0f}%  | J greedy={st_['J_greedy']:.3g} nearest={st_['J_nearest']:.3g}")
    prog.done()
    df = pd.DataFrame(rows)
    df.to_csv(ctx.path("single_cycle.csv"), index=False)
    ctx.finish(summary=dict(cells=len(cells)))


if __name__ == "__main__":
    main()
