# -*- coding: utf-8 -*-
"""
EXP-04  Steady-state probe: uniform radius vs lagged structure on a random network.

Long runs so that the post-burn-in mean is a steady state, and the transient
is reported separately. Defaults: random-fixed network observing 25% of the
interior, N = 40, 120 cycles, seeds 5000 and 5017, arms uniform r = 1, 2, 3
and lagged c = 3. Everything is overridable from the command line:

    python experiments/exp04_probe.py paper --network random-moving --N 80 \\
        --cycles 120 --seeds 5000,5017 --arms fixed1,fixed2,fixed3,lagged3

Output: results/EXP-04_<scale>/<tag>/probe.csv with, per run, the mean over
cycles >= burn-in (60), the mean of the last 40, the mean of the first 40
(transient), the spread, and divergence.
"""
from __future__ import annotations

import argparse, os, sys, time
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ExperimentContext, get_scale, make_testbed
from qgloc.filters import run_cycles
from qgloc.persist import RunRecorder
from qgloc.progress import Progress, log

EXP_ID = "EXP-04"


def parse_arm(a):
    if a.startswith("fixed"):
        return "enkf-mc", ("fixed", int(a[5:])), {}
    if a.startswith("letkf"):
        return "letkf", ("fixed", int(a[5:])), {}
    if a.startswith("lagged"):
        return "enkf-mc-lagged", ("lagged", None), dict(lag_c=float(a[6:]))
    if a.startswith("climstart"):          # climstartN: climate prior for N cycles, then uniform r=2
        return "enkf-mc-climstart", ("climstart", None), dict(climstart_cycles=int(a[9:] or 1), lasso_window=6, lasso_c=0.5)
    if a == "clim":                          # climate structure, ridge toward zero, all cycles
        return "enkf-mc-clim", ("clim", None), dict(lasso_window=6, lasso_c=0.5)
    raise ValueError(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", nargs="?", default=None)
    ap.add_argument("--network", default="random-fixed")
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=120)
    ap.add_argument("--burn-in", type=int, default=60)
    ap.add_argument("--seeds", default="5000,5017")
    ap.add_argument("--arms", default="fixed1,fixed2,climstart1,climstart2,clim")
    ap.add_argument("--inflation", type=float, default=None)
    ap.add_argument("--obs-freq", type=float, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    scale = get_scale(a.scale)
    tag = a.tag or f"{a.network}_{'s'+str(a.stride) if a.network=='lattice' else 'd'+str(int(100*a.density))}_N{a.N}_T{int(a.obs_freq or scale.obs_freq)}"
    ctx = ExperimentContext(EXP_ID, f"steady-state probe: {tag}", scale)
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = [parse_arm(x) for x in a.arms.split(",")]
    over = dict(ensemble_size=a.N, obs_density=a.density, obs_stride=a.stride)
    if a.inflation is not None:
        over["inflation"] = a.inflation
    if a.obs_freq is not None:
        over["obs_freq"] = a.obs_freq
    rows, metrics = [], []
    prog = Progress(len(seeds) * len(arms), label="runs", exp_id=EXP_ID)
    for seed in seeds:
        for filt, arm, o in arms:
            bed = make_testbed(scale, obs_network=a.network, **over, **o)
            X0, xt0 = bed.build_ensemble(seed)
            name = f"{arm[0]}{arm[1] if arm[1] else ''}" + (f"_c{o['lag_c']:g}" if 'lag_c' in o else "") + (f"_{o['climstart_cycles']}cyc" if 'climstart_cycles' in o else "") + ("" if filt != "letkf" else "_letkf")
            out = ctx.path(tag, name, f"seed{seed}", "run.npz")
            if os.path.exists(out):
                z = np.load(out, allow_pickle=True); rr = pd.DataFrame(z["metrics"]).to_dict("records")
            else:
                rec = RunRecorder(snap_cycles=(0, a.cycles // 2, a.cycles - 1), binv_cycles=(a.cycles // 2,))
                t0 = time.time()
                rr = run_cycles(bed, X0, xt0, filt, arm, seed=seed, network=a.network, recorder=rec, cycles=a.cycles)
                rec.write(out, dict(tag=tag, filt=filt, arm=arm, over=dict(over, **o), seed=seed,
                                    cfg=asdict(bed.cfg), elapsed_s=round(time.time() - t0, 1)), bed)
            ok = [r for r in rr if not r.get("diverged")]
            e = np.array([r["rmse_q"] for r in ok])
            s = dict(tag=tag, filt=filt, arm=name, seed=seed, n_cycles=len(ok),
                     diverged=len(ok) < a.cycles or bool(np.nanmax(e[-10:]) > 1.0),
                     rmse_steady=float(e[a.burn_in:].mean()) if len(e) > a.burn_in else np.nan,
                     rmse_last40=float(e[-40:].mean()), rmse_first40=float(e[:40].mean()),
                     cycles_below_0p3=int(np.argmax(e < 0.3)) if (e < 0.3).any() else -1,
                     spread_last40=float(np.mean([r["spread"] for r in ok[-40:]])),
                     pred_mean=float(np.nanmean([r.get("pred_mean", np.nan) for r in ok])) if filt != "enkf-mc" else np.nan)
            rows.append(s)
            for r in rr:
                metrics.append(dict(tag=tag, filt=filt, arm=name, seed=seed, **r))
            prog.step(f"{name}/seed{seed}: steady={s['rmse_steady']:.4f} last40={s['rmse_last40']:.4f} first40={s['rmse_first40']:.4f}" + (" DIVERGED" if s["diverged"] else ""))
            pd.DataFrame(rows).to_csv(ctx.path(tag, "probe.csv"), index=False)
            pd.DataFrame(metrics).to_csv(ctx.path(tag, "metrics.csv"), index=False)
    prog.done()
    df = pd.DataFrame(rows)
    g = df.groupby("arm").agg(steady=("rmse_steady", "mean"), last40=("rmse_last40", "mean"),
                              first40=("rmse_first40", "mean"), to_0p3=("cycles_below_0p3", "mean"),
                              spread=("spread_last40", "mean"), diverged=("diverged", "sum"))
    log("\n" + g.round(4).to_string(), EXP_ID)
    ctx.finish(summary=df.to_dict("records"))


if __name__ == "__main__":
    main()
