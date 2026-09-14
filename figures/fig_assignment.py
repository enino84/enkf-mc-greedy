# -*- coding: utf-8 -*-
"""Assignment figures: radius fields, tessellations, discarded observations,
sparsity of B^-1, the fields of a cycle. From EXP-01 (single forecasts) and
from the per-run archives of EXP-02 (inside the cycled filter).

    python figures/fig_assignment.py paper
"""
from __future__ import annotations

import glob, os, sys

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fig import (ARM_COLOR, ARM_LABEL, exp_dir, fig_dir, first_seed, grid_side,
                        interior_mask, label_arm, load_run, load_summary, main_arm,
                        method_arms, obs_rc, rule_arms, run_path, setup_matplotlib, to2d)


def _main_tanda(s, network=None):
    """The N tanda to draw single-run figures from: the smallest N, the middle density."""
    if network is not None:
        s = s[s.network == network]
    ts = sorted(t for t in s.tanda.unique() if t.startswith("N"))
    pref = [t for t in ts if t.endswith("_s2") or t.endswith("_d25")]
    return (pref or ts or [s.tanda.iloc[0]])[0]

RADIUS_CMAP = "viridis"


def _radius_norm(rmax):
    return BoundaryNorm(np.arange(-0.5, rmax + 1.5, 1), 256)


def _draw_radius(ax, rf, obs_idx, rmax, title, plt, obs_used=None):
    g = grid_side(rf.size)
    R = to2d(rf).astype(float)
    R[~interior_mask(g)] = np.nan
    im = ax.imshow(R, cmap=RADIUS_CMAP, norm=_radius_norm(rmax), origin="lower", interpolation="nearest")
    r, c = obs_rc(obs_idx, g)
    if obs_used is None:
        ax.scatter(c, r, s=6, c="white", edgecolors="k", linewidths=0.4)
    else:
        ax.scatter(c[obs_used], r[obs_used], s=7, c="white", edgecolors="k", linewidths=0.4)
        ax.scatter(c[~obs_used], r[~obs_used], s=16, marker="x", c="#ff2a2a", linewidths=0.9)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return im


def radius_fields(scale, plt, network="lattice", filt="enkf-mc", cycle=None):
    """The method's radius fields (one per rho) and a uniform field of the same mean."""
    seed = first_seed(scale); s = load_summary(scale); tanda = _main_tanda(s, network)
    arms = method_arms(s[(s.tanda == tanda) & (s.network == network) & (s.filt == filt)].arm.unique())
    runs = {}
    for arm in arms:
        p = run_path(scale, tanda, network, filt, arm, seed)
        if os.path.exists(p):
            runs[arm] = load_run(p)
    if not runs:
        return
    main = main_arm(runs)
    any_run = runs[main]
    cyc = int(any_run["snap_cycles"][-1]) if cycle is None else cycle
    cyc = min(cyc, any_run["radius"].shape[0] - 1)
    rmax = int(max(r["radius"][cyc].max() for r in runs.values()))
    ru = int(round(any_run["radius"][cyc][interior_mask(grid_side(any_run["radius"][cyc].size)).ravel()].mean()))
    rmax = max(rmax, ru)
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.2 * (len(runs) + 1), 3.4))
    for ax, arm in zip(axes, list(runs)):
        z = runs[arm]
        used = z["obs_used"][cyc] if "obs_used" in z else None
        rf = z["radius"][cyc]; rm = rf[interior_mask(grid_side(rf.size)).ravel()].mean()
        _draw_radius(ax, rf, z["obs_idx"][cyc], rmax, f"{label_arm(arm)}\nmean radius {rm:.2f}", plt, used)
    g = grid_side(any_run["radius"][cyc].size)
    rf_u = np.where(interior_mask(g).ravel(), ru, 0)
    im = _draw_radius(axes[-1], rf_u, any_run["obs_idx"][cyc], rmax, f"uniform radius {ru}\n(same mean as ρ={main[7:]})", plt)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, ticks=range(0, rmax + 1))
    cb.set_label("radius (0 = abstains)")
    fig.suptitle(f"radius fields at cycle {cyc}, {network} network, inside the {filt} — red × = observed site no component chose", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"radius_fields_{network}_{filt}.png"), dpi=160)
    fig.savefig(os.path.join(fig_dir(scale), f"radius_fields_{network}_{filt}.pdf"))
    plt.close(fig)


def radius_evolution(scale, plt, network="lattice", filt="enkf-mc"):
    seed = first_seed(scale); s = load_summary(scale); tanda = _main_tanda(s, network)
    main = main_arm(s[(s.tanda == tanda) & (s.network == network) & (s.filt == filt)].arm.unique())
    p = run_path(scale, tanda, network, filt, main, seed) if main else ""
    if not main or not os.path.exists(p):
        return
    z = load_run(p)
    cycles = [int(c) for c in z["snap_cycles"] if c < z["radius"].shape[0]]
    rmax = int(z["radius"].max())
    fig, axes = plt.subplots(1, len(cycles), figsize=(3.0 * len(cycles), 3.3))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cycles):
        used = z["obs_used"][c]
        im = _draw_radius(ax, z["radius"][c], z["obs_idx"][c], rmax,
                          f"cycle {c}: mean {z['radius'][c][interior_mask(grid_side(z['radius'][c].size)).ravel()].mean():.2f}, "
                          f"{int((~used).sum())} obs unused", plt, used)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, ticks=range(0, rmax + 1))
    cb.set_label("radius")
    fig.suptitle(f"{label_arm(main)} along the {filt} run, {network} network", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"radius_evolution_{network}_{filt}.png"), dpi=160)
    plt.close(fig)


def _tess_colors(n):
    rng = np.random.default_rng(3)
    base = plt_cmap = None
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    cols = np.array([cm(i % 20) for i in rng.permutation(n)])
    return ListedColormap(cols)


def tessellation(scale, plt, network="lattice", filt="enkf-mc"):
    """Which observation updates each component: greedy against nearest."""
    seed = first_seed(scale); s = load_summary(scale); tanda = _main_tanda(s, network)
    main = main_arm(s[(s.tanda == tanda) & (s.network == network) & (s.filt == filt)].arm.unique())
    if not main or not os.path.exists(run_path(scale, tanda, network, filt, main, seed)):
        return
    runs = {"greedy": load_run(run_path(scale, tanda, network, filt, main, seed))}
    z = runs["greedy"]
    # the nearest-observation tessellation, from the observation positions
    g0 = grid_side(z["assigned_obs"][0].size)
    def nearest_of(obs_idx):
        rr, cc = np.divmod(obs_idx, g0); ii = np.arange(g0 * g0); ri, ci = np.divmod(ii, g0)
        d = np.maximum(np.abs(ri[:, None] - rr[None, :]), np.abs(ci[:, None] - cc[None, :]))
        return d.argmin(axis=1)
    runs["nearest"] = dict(assigned_obs=np.stack([nearest_of(o) for o in z["obs_idx"]]),
                           obs_used=np.ones_like(z["obs_used"]), obs_idx=z["obs_idx"], snap_cycles=z["snap_cycles"])
    cyc = int(min(z["snap_cycles"][-1], z["assigned_obs"].shape[0] - 1))
    obs_idx = z["obs_idx"][cyc]; g = grid_side(z["assigned_obs"][cyc].size)
    n_obs = obs_idx.size
    cmap = _tess_colors(n_obs)
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.6 * (len(runs) + 1), 3.6))
    for ax, arm in zip(axes, ["nearest", "greedy"]):
        if arm not in runs:
            continue
        A = to2d(runs[arm]["assigned_obs"][cyc]).astype(float)
        A[~interior_mask(g)] = np.nan
        A[A < 0] = np.nan
        ax.imshow(A, cmap=cmap, vmin=-0.5, vmax=n_obs - 0.5, origin="lower", interpolation="nearest")
        r, c = obs_rc(obs_idx, g)
        used = runs[arm]["obs_used"][cyc]
        ax.scatter(c[used], r[used], s=8, c="white", edgecolors="k", linewidths=0.5)
        ax.scatter(c[~used], r[~used], s=22, marker="x", c="k", linewidths=1.0)
        ax.set_title(label_arm(main) if arm == "greedy" else ARM_LABEL["nearest"])
        ax.set_xticks([]); ax.set_yticks([])
    # difference map
    if "nearest" in runs:
        D = (runs["greedy"]["assigned_obs"][cyc] != runs["nearest"]["assigned_obs"][cyc]).astype(float)
        D2 = to2d(D); D2[~interior_mask(g)] = np.nan
        axes[-1].imshow(D2, cmap=ListedColormap(["#f0f0f0", ARM_COLOR["greedy"]]), vmin=0, vmax=1, origin="lower", interpolation="nearest")
        r, c = obs_rc(obs_idx, g)
        axes[-1].scatter(c, r, s=6, c="white", edgecolors="k", linewidths=0.4)
        frac = D[interior_mask(g).ravel()].mean()
        axes[-1].set_title(f"method ≠ nearest: {100*frac:.0f}% of the interior")
        axes[-1].set_xticks([]); axes[-1].set_yticks([])
    fig.suptitle(f"which observed site sets each point's radius — cycle {cyc}, {network} network, {filt}; × = site no component chose", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"tessellation_{network}_{filt}.png"), dpi=160)
    fig.savefig(os.path.join(fig_dir(scale), f"tessellation_{network}_{filt}.pdf"))
    plt.close(fig)


def discarded_obs(scale, plt, network="lattice", filt="enkf-mc"):
    """Ensemble spread with the observations that update nothing marked."""
    seed = first_seed(scale); s = load_summary(scale); tanda = _main_tanda(s, network)
    main = main_arm(s[(s.tanda == tanda) & (s.network == network) & (s.filt == filt)].arm.unique())
    p = run_path(scale, tanda, network, filt, main, seed) if main else ""
    if not main or not os.path.exists(p):
        return
    z = load_run(p)
    sc = [int(c) for c in z["snap_cycles"] if c < z["obs_used"].shape[0]]
    fig, axes = plt.subplots(1, len(sc) + 1, figsize=(3.2 * (len(sc) + 1), 3.4),
                             gridspec_kw=dict(width_ratios=[1] * len(sc) + [0.9]))
    for ax, c in zip(axes[:-1], sc):
        j = list(z["snap_cycles"]).index(c)
        S = to2d(z["spread_q"][j]); g = S.shape[0]; S[~interior_mask(g)] = np.nan
        im = ax.imshow(np.log10(S + 1e-12), cmap="magma", origin="lower", interpolation="nearest")
        r, cc = obs_rc(z["obs_idx"][c], g); used = z["obs_used"][c]
        ax.scatter(cc[used], r[used], s=7, c="white", edgecolors="k", linewidths=0.4)
        ax.scatter(cc[~used], r[~used], s=20, marker="x", c="#00e5ff", linewidths=1.0)
        ax.set_title(f"cycle {c}: {int((~used).sum())} of {used.size} unused"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:-1].tolist(), fraction=0.02, pad=0.02, label="log10 ensemble spread of q")
    # variance at used vs unused sites, all cycles
    vu, vn = [], []
    for c in range(z["obs_used"].shape[0]):
        used = z["obs_used"][c]; v = z["obs_var_ens"][c]
        vu.extend(v[used]); vn.extend(v[~used])
    ax = axes[-1]
    bins = np.logspace(np.log10(max(1e-12, min(vu + vn))), np.log10(max(vu + vn) + 1e-12), 30)
    ax.hist(vu, bins=bins, color="#4d4d4d", alpha=0.7, label=f"used (median {np.median(vu):.2e})")
    if vn:
        ax.hist(vn, bins=bins, color="#00b8d4", alpha=0.8, label=f"unused (median {np.median(vn):.2e})")
    ax.set_xscale("log"); ax.set_xlabel("ensemble variance at the observation site"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("all cycles")
    fig.suptitle(f"observed sites no component chose ({label_arm(main)}), {network} network, {filt}", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"discarded_obs_{network}_{filt}.png"), dpi=160)
    plt.close(fig)


def sparsity(scale, plt, network="lattice"):
    """spy(B^-1) for the four radius fields, from EXP-01 (which has the uniform)."""
    d = exp_dir("EXP-01", scale)
    files = sorted(glob.glob(os.path.join(d, network, "*", "seed*", "cycle*.npz")))
    if not files:
        return
    z = np.load(files[-1])
    n = z["radius_greedy"].size
    arms = [k[7:] for k in z.files if k.startswith("radius_partial")] + ["uniform", "nearest", "greedy"]
    fig, axes = plt.subplots(2, len(arms), figsize=(3.3 * len(arms), 6.6), squeeze=False)
    zoom = slice(0, min(n, 400))
    for j, arm in enumerate(arms):
        ip, ix = z[f"binv_{arm}_indptr"], z[f"binv_{arm}_indices"]
        rows = np.repeat(np.arange(n), np.diff(ip))
        nnz = ix.size
        for ax, (lo, hi) in zip(axes[:, j], [(0, n), (zoom.start, zoom.stop)]):
            m = (rows >= lo) & (rows < hi) & (ix >= lo) & (ix < hi)
            ax.scatter(ix[m], rows[m], s=0.15 if hi - lo > 500 else 2, c="k", marker="s", lw=0)
            ax.set_xlim(lo, hi); ax.set_ylim(hi, lo); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
        rf = z[f"radius_{arm}"]; g = grid_side(n)
        axes[0, j].set_title(f"{label_arm(arm)}\nmean r {rf[interior_mask(g).ravel()].mean():.2f}, {100*nnz/n**2:.2f}% nonzero")
        axes[1, j].set_title(f"first {zoom.stop} components")
    fig.suptitle(f"sparsity of $B^{{-1}}$ under each radius field ({network} network, {os.path.basename(files[-1])})", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"sparsity_{network}.png"), dpi=160)
    plt.close(fig)


def fields(scale, plt, network="lattice", filt="enkf-mc"):
    """Truth, background, analysis and errors at the last snapshot cycle, per arm."""
    seed = first_seed(scale)
    s = load_summary(scale); tanda = _main_tanda(s, network)
    s = s[(s.tanda == tanda) & (s.network == network) & (s.filt == filt) & (s.seed == seed)]
    fixed = s[s.kind == "fixed"].sort_values("rmse")
    arms = ([fixed.arm.iloc[0]] if len(fixed) else []) + method_arms(s.arm.unique())
    runs = {a: load_run(run_path(scale, tanda, network, filt, a, seed)) for a in arms}
    runs = {a: z for a, z in runs.items() if len(z["snap_cycles"])}
    if not runs:
        return
    cyc = int(min(min(z["snap_cycles"][-1] for z in runs.values()), 10 ** 6))
    fig, axes = plt.subplots(len(runs), 4, figsize=(12.5, 3.0 * len(runs)), squeeze=False)
    for i, (arm, z) in enumerate(runs.items()):
        j = list(z["snap_cycles"]).index(cyc) if cyc in list(z["snap_cycles"]) else len(z["snap_cycles"]) - 1
        xt, xb, xa = to2d(z["xt_q"][j]), to2d(z["xb_q"][j]), to2d(z["xa_q"][j])
        v = np.nanpercentile(np.abs(xt), 99)
        for ax, (F, t) in zip(axes[i, :3], [(xt, "truth"), (xb, "background mean"), (xa, "analysis mean")]):
            ax.imshow(F, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{t}" if i == 0 else "")
        e = np.abs(xa - xt); ev = np.nanpercentile(np.abs(xb - xt), 99)
        axes[i, 3].imshow(e, cmap="magma", vmin=0, vmax=ev, origin="lower"); axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])
        axes[i, 3].set_title("|analysis − truth|" if i == 0 else "")
        rm = z["metrics"]["rmse_q"][int(z["snap_cycles"][j])] if "rmse_q" in z["metrics"].dtype.names else np.nan
        axes[i, 0].set_ylabel(f"{label_arm(arm)}\nRMSE {rm:.3f}", fontsize=8)
    fig.suptitle(f"q at cycle {cyc}, {network} network, {filt} (best uniform radius on top)", fontsize=9)
    fig.savefig(os.path.join(fig_dir(scale), f"fields_{network}_{filt}.png"), dpi=140)
    plt.close(fig)


def single_cycle_costs(scale, plt):
    """From EXP-01: J per rule, and the radius histogram, both networks."""
    import pandas as pd
    p = os.path.join(exp_dir("EXP-01", scale), "single_cycle.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    nets = sorted(df.network.unique())
    fig, axes = plt.subplots(1, 2 * len(nets), figsize=(3.6 * 2 * len(nets), 3.0))
    axes = np.atleast_1d(axes)
    for k, net in enumerate(nets):
        sub = df[df.network == net]
        ax = axes[2 * k]
        keys = ["J_abstain", "J_nearest", "J_random", "J_greedy"]; labels = ["abstain", "nearest", "random", "greedy"]
        med = [sub[c].median() for c in keys]; lo = [sub[c].min() for c in keys]; hi = [sub[c].max() for c in keys]
        cols = ["#bbbbbb", ARM_COLOR["nearest"], ARM_COLOR["random"], ARM_COLOR["greedy"]]
        ax.bar(labels, med, color=cols, yerr=[np.subtract(med, lo), np.subtract(hi, med)], capsize=3)
        ax.set_yscale("log"); ax.set_ylabel("J (mean local cost)"); ax.set_title(f"{net} network: cost of each rule")
        ax = axes[2 * k + 1]
        files = sorted(glob.glob(os.path.join(exp_dir("EXP-01", scale), net, "*", "seed*", "cycle*.npz")))
        if files:
            z = np.load(files[-1]); g = grid_side(z["radius_greedy"].size); m = interior_mask(g).ravel()
            parts = [k[7:] for k in z.files if k.startswith("radius_partial")]
            rmax = int(max(z[f"radius_{a}"].max() for a in parts + ["greedy", "nearest"]))
            from common_fig import color_arm
            for arm in parts + ["greedy", "nearest"]:
                ax.hist(z[f"radius_{arm}"][m], bins=np.arange(-0.5, rmax + 1.5), histtype="step", lw=1.6,
                        color=color_arm(arm), label=label_arm(arm))
            ax.set_xlabel("radius"); ax.set_ylabel("components"); ax.set_title("radius distribution"); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir(scale), "single_cycle_costs.png"), dpi=150)
    plt.close(fig)


def main(scale):
    plt = setup_matplotlib()
    s = load_summary(scale)
    for net in sorted(s.network.unique()):
        for filt in ("enkf-mc", "letkf"):
            radius_fields(scale, plt, net, filt)
            radius_evolution(scale, plt, net, filt)
            tessellation(scale, plt, net, filt)
            discarded_obs(scale, plt, net, filt)
            fields(scale, plt, net, filt)
        sparsity(scale, plt, net)
    single_cycle_costs(scale, plt)
    print("assignment figures ->", fig_dir(scale))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCALE", "smoke"))
