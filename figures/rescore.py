# -*- coding: utf-8 -*-
"""Re-score a finished EXP-02 with a different cycle window. No run is repeated.

    python figures/rescore.py paper --burn-in 20            # cycles 20..end
    python figures/rescore.py paper --burn-in 15 --last 45  # cycles 15..45

Reads metrics.csv (every cycle of every run), rewrites summary.csv, and keeps
the previous one as summary_prev.csv. Then rerun fig_benchmark.py and
make_tables.py as usual.
"""
from __future__ import annotations

import argparse, os, shutil, sys

import numpy as np, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fig import exp_dir, load_metrics

TAGS = ["tanda", "network", "filt", "arm", "kind", "radius", "rho", "seed", "N",
        "alpha", "stride", "density"]
MEANS = {"rmse": "rmse_q", "b_rmse": "b_rmse_q", "spread": "spread", "r_mean": "r_mean",
         "t_analysis": "t_analysis", "t_assign": "t_assign", "J": "J",
         "frac_not_nearest": "frac_not_nearest", "frac_abstain": "frac_abstain",
         "obs_unused": "obs_unused"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale")
    ap.add_argument("--burn-in", type=int, required=True, help="first cycle scored (inclusive)")
    ap.add_argument("--last", type=int, default=None, help="last cycle scored (inclusive); default: end")
    a = ap.parse_args()
    d = exp_dir("EXP-02", a.scale)
    m = load_metrics(a.scale)
    tags = [t for t in TAGS if t in m.columns]
    rows = []
    for key, g in m.groupby(tags, sort=False):
        g = g.sort_values("cycle")
        diverged = bool(g["diverged"].astype(bool).any())
        div_at = int(g.loc[g["diverged"].astype(bool), "cycle"].min()) if diverged else -1
        w = g[(g.cycle >= a.burn_in) & ((g.cycle <= a.last) if a.last is not None else True)
              & ~g["diverged"].astype(bool)]
        s = dict(zip(tags, key if isinstance(key, tuple) else (key,)))
        for out, col in MEANS.items():
            s[out] = float("nan") if diverged or col not in w or w.empty else float(np.nanmean(w[col]))
        s.update(diverged=diverged, diverged_at=div_at, n_cycles=int(len(g)),
                 score_from=a.burn_in, score_to=a.last if a.last is not None else int(g.cycle.max()))
        rows.append(s)
    out = pd.DataFrame(rows)
    prev = os.path.join(d, "summary.csv")
    if os.path.exists(prev):
        shutil.copy(prev, os.path.join(d, "summary_prev.csv"))
    out.to_csv(prev, index=False)
    print(f"rescored {len(out)} runs, cycles {a.burn_in}..{a.last if a.last is not None else 'end'} -> {prev}")


if __name__ == "__main__":
    main()
