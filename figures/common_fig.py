# -*- coding: utf-8 -*-
"""Shared loading and styling for the figure and table scripts.

Every script here reads only what EXP-01 and EXP-02 wrote to disk. Nothing is
recomputed from the model, so a figure can be regenerated in seconds from a
finished results directory, on any machine, without pyteda.
"""
from __future__ import annotations

import glob, json, os, sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("RESULTS_DIR", os.path.join(REPO_ROOT, "results"))

# The method arms are "partial<rho>"; the candidate-based rules of the first
# draft are kept for the legacy/oneobs diagnostics.
RULE_ORDER = ["greedy", "nearest", "random", "variance"]
ARM_LABEL = dict(greedy="assignment by J (first draft)", nearest="nearest observation",
                 random="random candidate", variance="raw-variance rule", uniform="uniform radius")
ARM_COLOR = dict(greedy="#7b3294", nearest="#2166ac", random="#4d4d4d", variance="#1b9e77", uniform="#e08214")
PARTIAL_COLORS = {3: "#ef8a62", 4: "#b2182b", 5: "#67001f", 2: "#f4a582"}
FILT_LABEL = {"enkf-mc": "EnKF-MC", "letkf": "LETKF", "enkf-mc-bayes": "EnKF-MC Bayesian rows", "enkf-mc-masked": "EnKF-MC masked",
              "enkf-mc-group": "EnKF-MC group", "letkf-only": "LETKF-only"}


def is_partial(arm):
    return str(arm).startswith("partial") or str(arm) == "bayes"


def method_arms(arms):
    """The partial arms of a list, sorted by rho."""
    return sorted([a for a in arms if is_partial(a)], key=lambda a: (a != "bayes", a))


def main_arm(arms):
    """The method arm to draw when only one is drawn: rho = 4 if present."""
    m = method_arms(arms)
    return "bayes" if "bayes" in m else ("partial4" if "partial4" in m else (m[0] if m else None))


def rule_arms(arms):
    return [a for a in RULE_ORDER if a in set(arms)]


ARM_ORDER = RULE_ORDER


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": False,
                         "axes.titlesize": 9.5, "legend.fontsize": 8,
                         "mathtext.fontset": "cm", "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def exp_dir(exp_id, scale):
    return os.path.join(RESULTS_ROOT, f"{exp_id}_{scale}")


def fig_dir(scale):
    d = os.path.join(RESULTS_ROOT, f"figures_{scale}")
    os.makedirs(d, exist_ok=True)
    return d


def load_csv_shards(d, stem):
    """``stem.csv`` or the union of ``stem_shard*.csv``."""
    one = os.path.join(d, f"{stem}.csv")
    parts = sorted(glob.glob(os.path.join(d, f"{stem}_shard*.csv")))
    frames = [pd.read_csv(p) for p in ([one] if os.path.exists(one) else []) + parts]
    if not frames:
        raise FileNotFoundError(f"no {stem} csv under {d}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=[c for c in ("tanda", "network", "filt", "arm", "seed", "cycle") if c in df.columns])


def load_summary(scale):
    return load_csv_shards(exp_dir("EXP-02", scale), "summary")


def load_metrics(scale):
    return load_csv_shards(exp_dir("EXP-02", scale), "metrics")


def run_path(scale, tanda, network, filt, arm, seed):
    return os.path.join(exp_dir("EXP-02", scale), tanda, network, filt, arm, f"seed{seed}", "run.npz")


def load_run(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    cfg_path = os.path.splitext(path)[0] + "_config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            d["config"] = json.load(fh)
    return d


def first_seed(scale):
    s = load_summary(scale)
    return int(sorted(s["seed"].unique())[0])


def grid_side(nq):
    return int(round(np.sqrt(nq)))


def to2d(v, fill=np.nan):
    v = np.asarray(v, dtype=float)
    g = grid_side(v.size)
    return v.reshape(g, g)


def interior_mask(g):
    m = np.ones((g, g), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m


def obs_rc(obs_idx, g):
    return np.divmod(np.asarray(obs_idx), g)


def label_arm(arm):
    if arm.startswith("fixed"):
        return f"uniform r={arm[5:]}"
    if arm == "bayes":
        return "EnKF-MC, Bayesian rows"
    if is_partial(arm):
        return f"partial-corr. radii, ρ={arm[7:]}"
    return ARM_LABEL.get(arm, arm)


def color_arm(arm, cmap=None):
    if arm.startswith("fixed"):
        import matplotlib.pyplot as plt
        r = int(arm[5:])
        return plt.get_cmap("YlOrBr")(0.3 + 0.08 * r)
    if arm == "bayes":
        return "#b2182b"
    if is_partial(arm):
        return PARTIAL_COLORS.get(int(arm[7:]), "#b2182b")
    return ARM_COLOR.get(arm, "k")


def median_band(df, by, key="rmse"):
    g = df.groupby(by)[key]
    return g.median(), g.min(), g.max()
