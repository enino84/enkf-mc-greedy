# -*- coding: utf-8 -*-
"""
EXP-03  Predecessors from the estimated propagator (enkf-mc-lagged) vs uniform.

The lagged cross-correlation between the previous analysis ensemble and the
forecast ensemble estimates the propagator: which points fed each component
during the interval. At N = 40 the structure it gives (threshold 3/sqrt(N))
converges over 36 cycles to 0.16 against 0.11 for the best uniform radius;
its errors are false positives of the selection, which shrink with N. This
runs uniform r = 2, 3, 4 against lagged with c in {2.5, 3, 3.5} at N = 40,
80 and 120, fixed lattice stride 2, inflation 1.05, 60 cycles, one seed.

Writes results/EXP-03_<scale>/flow_probe.csv (one row per run, post-burn-in
RMSE) and metrics_flow_probe.csv (every cycle). About 2-3 h on one core.

    SCALE=paper docker compose run -d exp03
"""
from __future__ import annotations

import os, sys, time
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ExperimentContext, get_scale, make_testbed, parse_cli
from qgloc.filters import run_cycles, score
from qgloc.persist import RunRecorder
from qgloc.progress import Progress, log

EXP_ID = "EXP-03"
ARMS = [("enkf-mc", ("fixed", 2), {}), ("enkf-mc", ("fixed", 3), {}), ("enkf-mc", ("fixed", 4), {}),
        ("enkf-mc-lagged", ("lagged", None), dict(lag_c=2.5)),
        ("enkf-mc-lagged", ("lagged", None), dict(lag_c=3.0)),
        ("enkf-mc-lagged", ("lagged", None), dict(lag_c=3.5))]


def main():
    a = parse_cli()
    scale = get_scale(a.scale)
    ctx = ExperimentContext(EXP_ID, "lagged-propagator precision vs uniform, N = 40/80/120", scale)
    Ns = (40, 80, 120) if scale.name.startswith("paper") else (scale.ensemble_sizes[0],)
    cycles = scale.cycles
    seed = scale.seeds[0]
    rows, metrics = [], []
    prog = Progress(len(Ns) * len(ARMS), label="runs", exp_id=EXP_ID)
    for N in Ns:
        for filt, arm, over in ARMS:
            bed = make_testbed(scale, obs_network="lattice", ensemble_size=N, obs_stride=2, **over)
            X0, xt0 = bed.build_ensemble(seed)
            name = f"{filt}_{arm[0]}{arm[1] if arm[1] else ''}" + (f"_c{over['lag_c']}" if over else "")
            out = ctx.path(f"N{N}", name, "run.npz")
            if os.path.exists(out):
                z = np.load(out, allow_pickle=True); rr = pd.DataFrame(z["metrics"]).to_dict("records")
            else:
                rec = RunRecorder(snap_cycles=(0, cycles // 2, cycles - 1), binv_cycles=(cycles // 2,))
                t0 = time.time()
                rr = run_cycles(bed, X0, xt0, filt, arm, seed=seed, network="lattice", recorder=rec, cycles=cycles)
                rec.write(out, dict(N=N, filt=filt, arm=arm, over=over, cfg=asdict(bed.cfg),
                                    elapsed_s=round(time.time() - t0, 1)), bed)
            tags = dict(N=N, filt=filt, arm=name)
            for r in rr:
                metrics.append(dict(tags, **r))
            s = dict(tags, rmse=score(rr, scale.burn_in, "rmse_q"), rmse_last10=float(np.mean([r["rmse_q"] for r in rr[-10:] if not r.get("diverged")] or [np.nan])),
                     spread=score(rr, scale.burn_in, "spread"), diverged=any(r.get("diverged") for r in rr),
                     pred_mean=score(rr, scale.burn_in, "pred_mean") if filt != "enkf-mc" else np.nan)
            rows.append(s)
            prog.step(f"N{N}/{name}: rmse={s['rmse']:.4f} last10={s['rmse_last10']:.4f}" + (" DIVERGED" if s["diverged"] else ""))
            pd.DataFrame(rows).to_csv(ctx.path("flow_probe.csv"), index=False)
            pd.DataFrame(metrics).to_csv(ctx.path("metrics_flow_probe.csv"), index=False)
    prog.done()
    df = pd.DataFrame(rows)
    log("\n" + df[["N", "arm", "rmse", "rmse_last10", "spread", "pred_mean", "diverged"]].round(4).to_string(index=False), EXP_ID)
    ctx.finish(summary=df.to_dict("records"))


if __name__ == "__main__":
    main()
