# -*- coding: utf-8 -*-
"""
EXP-02  Cycled benchmark: fixed radius against observation assignment.

Main tandas ``N40``, ``N80``, ``N120``, each with both filters and both
networks:

    fixed r        r in scale.radii    the uniform-radius baseline
    partial rho    rho in scale.rhos   the method: radii from the partial
                                       correlations of a probe precision

Diagnostic tandas on the lattice at N = 40: ``oneobs`` (the same radii but
each component updated by its assigned observation only), ``legacy`` (the
candidate-based rules of the first draft in the global filter), ``alpha``
and ``sparse`` (sensitivity).

Every run writes ``run.npz`` and ``run_config.json`` under
``results/EXP-02_<scale>/<tanda>/<network>/<filter>/<arm>/seed<seed>/`` with
everything the figures need (see ``qgloc/persist.py``). ``metrics.csv`` at the
top collects the per-cycle rows of every run; ``summary.csv`` the post-burn-in
scores. Shards write ``metrics_shard<i>.csv`` and are merged by
``figures/collect.py``.
"""
from __future__ import annotations

import os, sys, time
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SHARD_COUNT, SHARD_INDEX, ExperimentContext, get_scale,
                    make_testbed, parse_cli, shard_cells)
from qgloc.filters import run_cycles, score
from qgloc.persist import RunRecorder
from qgloc.progress import Progress, log

EXP_ID = "EXP-02"


def arm_name(arm):
    kind, r = arm
    if kind == "fixed":
        return f"fixed{int(r)}"
    if kind == "partial":
        return f"partial{int(r)}"
    return kind          # "flow"


def cells_for(scale, only=None):
    cells = []
    for tanda, over, filt_arms, networks in scale.tandas():
        if only and tanda != only:
            continue
        for net in networks:
            for filt, arms in filt_arms:
                for arm in arms:
                    for seed in scale.seeds:
                        cells.append(dict(tanda=tanda, over=over, network=net,
                                          filt=filt, arm=arm, seed=seed))
    return cells


def main():
    a = parse_cli()
    scale = get_scale(a.scale)
    ctx = ExperimentContext(EXP_ID, "cycled benchmark: fixed radius vs assignment", scale)
    cells = shard_cells(cells_for(scale, only=a.tanda))
    log(f"{len(cells)} runs in this shard ({SHARD_INDEX + 1}/{SHARD_COUNT})", EXP_ID)

    beds = {}
    metrics, summary = [], []
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID)
    for c in cells:
        key = (c["tanda"], c["network"])
        if key not in beds:
            beds[key] = make_testbed(scale, obs_network=c["network"], **c["over"])
        bed = beds[key]
        cfg = bed.cfg
        name = arm_name(c["arm"])
        out_dir = ctx.path(c["tanda"], c["network"], c["filt"], name, f"seed{c['seed']}", "run.npz")
        kind = c["arm"][0]
        tags = dict(tanda=c["tanda"], network=c["network"], filt=c["filt"],
                    arm=name, kind=kind, radius=c["arm"][1] if kind == "fixed" else -1,
                    rho=c["arm"][1] if kind == "partial" else -1,
                    seed=c["seed"], N=cfg.ensemble_size, alpha=cfg.ridge_alpha,
                    stride=cfg.obs_stride if c["network"] == "lattice" else -1,
                    density=cfg.obs_density if c["network"] != "lattice" else -1.0)
        if os.path.exists(out_dir):
            log(f"skip (exists) {tags}", EXP_ID)
            z = np.load(out_dir, allow_pickle=True)
            rows = pd.DataFrame(z["metrics"]).to_dict("records")
        else:
            X0, xt0 = bed.build_ensemble(c["seed"])
            rec = RunRecorder(snap_cycles=scale.snapshot_cycles,
                              binv_cycles=scale.snapshot_cycles,
                              keep_table_at=scale.snapshot_cycles)
            t0 = time.time()
            rows = run_cycles(bed, X0, xt0, c["filt"], c["arm"], seed=c["seed"],
                              network=c["network"], recorder=rec)
            config = dict(tags, cfg=asdict(cfg), scale=scale.name,
                          elapsed_s=round(time.time() - t0, 1),
                          snapshot_cycles=list(scale.snapshot_cycles))
            rec.write(out_dir, config, bed)
        for r in rows:
            metrics.append(dict(tags, **r))
        s = dict(tags, rmse=score(rows, cfg.burn_in, "rmse_q"),
                 rmse_raw=score(rows, cfg.burn_in, "rmse_q_raw"),
                 b_rmse=score(rows, cfg.burn_in, "b_rmse_q"),
                 spread_q_clim=float(bed.spread["q"]), obs_std_raw=float(cfg.obs_std * bed.spread["q"]),
                 spread=score(rows, cfg.burn_in, "spread"),
                 r_mean=score(rows, cfg.burn_in, "r_mean"),
                 t_analysis=score(rows, cfg.burn_in, "t_analysis"),
                 t_assign=score(rows, cfg.burn_in, "t_assign"),
                 diverged=any(r.get("diverged") for r in rows),
                 diverged_at=next((r["cycle"] for r in rows if r.get("diverged")), -1),
                 n_cycles=len(rows))
        for k in ("J", "frac_not_nearest", "frac_abstain", "obs_unused"):
            s[k] = score(rows, cfg.burn_in, k)
        summary.append(s)
        prog.step(f"{c['tanda']}/{c['network']}/{c['filt']}/{name}/s{c['seed']}: "
                  f"rmse={s['rmse']:.4f}" + (" DIVERGED" if s["diverged"] else ""))
    prog.done()

    suffix = "" if SHARD_COUNT == 1 else f"_shard{SHARD_INDEX}"
    pd.DataFrame(metrics).to_csv(ctx.path(f"metrics{suffix}.csv"), index=False)
    pd.DataFrame(summary).to_csv(ctx.path(f"summary{suffix}.csv"), index=False)
    ctx.finish(summary=dict(runs=len(cells),
                            diverged=int(sum(s["diverged"] for s in summary))))


if __name__ == "__main__":
    main()
