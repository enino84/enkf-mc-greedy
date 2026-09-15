# -*- coding: utf-8 -*-
"""Benchmark figures from EXP-02: error curves, error vs radius, divergence map.

    python figures/fig_benchmark.py paper
"""
from __future__ import annotations

import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fig import (FILT_LABEL, color_arm, fig_dir, label_arm, load_metrics,
                        load_summary, method_arms, rule_arms, setup_matplotlib)


def rmse_vs_cycle(scale, plt, tanda="main"):
    m = load_metrics(scale)
    m = m[(m.tanda == tanda) & (~m.diverged.astype(bool))].copy()
    m.loc[m.filt == "enkf-mc-bayes", "filt"] = "enkf-mc"        # the method is drawn in the EnKF-MC panel
    nets = sorted(m.network.unique()); filts = [f for f in ("enkf-mc", "letkf") if f in set(m.filt)]
    fig, axes = plt.subplots(len(nets), len(filts), figsize=(4.2 * len(filts), 3.0 * len(nets)),
                             squeeze=False, sharey="row")
    for i, net in enumerate(nets):
        for j, filt in enumerate(filts):
            ax = axes[i, j]
            sub = m[(m.network == net) & (m.filt == filt)]
            arms = sorted([a for a in sub.arm.unique() if a.startswith("fixed")], key=lambda a: int(a[5:])) \
                + method_arms(sub.arm.unique()) + rule_arms(sub.arm.unique())
            for arm in arms:
                s = sub[sub.arm == arm].groupby("cycle")["rmse_q"]
                med, lo, hi = s.median(), s.min(), s.max()
                lw = 1.8 if not arm.startswith("fixed") else 1.0
                ax.plot(med.index, med.values, color=color_arm(arm), lw=lw, label=label_arm(arm))
                if not arm.startswith("fixed"):
                    ax.fill_between(med.index, lo.values, hi.values, color=color_arm(arm), alpha=0.12, lw=0)
            ax.set_title(f"{FILT_LABEL[filt]}, {net} network")
            ax.set_xlabel("cycle"); ax.set_ylabel("analysis RMSE in q (normalized)")
            ax.grid(True, alpha=0.25)
    axes[0, -1].legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir(scale), f"rmse_vs_cycle_{tanda}.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir(scale), f"rmse_vs_cycle_{tanda}.pdf"))
    plt.close(fig)


def rmse_vs_radius(scale, plt, tanda="main"):
    s = load_summary(scale)
    s = s[s.tanda == tanda].copy()
    s.loc[s.filt == "enkf-mc-bayes", "filt"] = "enkf-mc"
    nets = sorted(s.network.unique()); filts = [f for f in ("enkf-mc", "letkf") if f in set(s.filt)]
    fig, axes = plt.subplots(len(nets), len(filts), figsize=(4.2 * len(filts), 3.0 * len(nets)),
                             squeeze=False, sharey="row")
    for i, net in enumerate(nets):
        for j, filt in enumerate(filts):
            ax = axes[i, j]
            sub = s[(s.network == net) & (s.filt == filt)]
            fx = sub[sub.kind == "fixed"].groupby("radius")["rmse"]
            med, lo, hi = fx.median(), fx.min(), fx.max()
            ax.errorbar(med.index, med.values, yerr=[med.values - lo.values, hi.values - med.values],
                        fmt="o-", color=color_arm("fixed4"), ms=4, lw=1.2, capsize=2, label="uniform radius")
            div = sub[(sub.kind == "fixed") & sub.diverged.astype(bool)]
            if len(div):
                ax.scatter(div.radius, np.full(len(div), np.nanmax(med.values)), marker="x",
                           color="k", s=30, zorder=5, label="diverged (some seed)")
            for arm in method_arms(sub.arm.unique()) + rule_arms(sub.arm.unique()):
                r = sub[sub.arm == arm]
                if r.empty:
                    continue
                v = r.rmse
                ax.axhline(v.median(), color=color_arm(arm), lw=1.8, label=label_arm(arm))
                ax.axhspan(v.min(), v.max(), color=color_arm(arm), alpha=0.10, lw=0)
                rm = r.r_mean.median()
                ax.plot([rm], [v.median()], marker="D", color=color_arm(arm), ms=5)
            ax.set_title(f"{FILT_LABEL[filt]}, {net} network")
            ax.set_xlabel("uniform radius (diamond: mean radius of the rule)")
            ax.set_ylabel("post-burn-in analysis RMSE in q")
            ax.grid(True, alpha=0.25)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir(scale), f"rmse_vs_radius_{tanda}.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir(scale), f"rmse_vs_radius_{tanda}.pdf"))
    plt.close(fig)


def divergence_map(scale, plt, tanda="main"):
    s = load_summary(scale)
    s = s[s.tanda == tanda].copy()
    s["cell"] = s.network.astype(str) + " / " + s.filt.map(lambda f: FILT_LABEL.get(f, f))
    arms = sorted([a for a in s.arm.unique() if a.startswith("fixed")], key=lambda a: int(a[5:])) \
        + method_arms(s.arm.unique()) + rule_arms(s.arm.unique())
    cells = sorted(s.cell.unique())
    M = np.full((len(cells), len(arms)), np.nan)
    for i, c in enumerate(cells):
        for j, a in enumerate(arms):
            r = s[(s.cell == c) & (s.arm == a)]
            if len(r):
                M[i, j] = r.diverged.astype(bool).mean()
    fig, ax = plt.subplots(figsize=(0.55 * len(arms) + 2, 0.45 * len(cells) + 1.2))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([label_arm(a) for a in arms], rotation=60, ha="right")
    ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells)
    for i in range(len(cells)):
        for j in range(len(arms)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100*M[i,j]:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.5 else "black")
    ax.set_title("fraction of seeds that diverged")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir(scale), f"divergence_{tanda}.png"), dpi=150)
    plt.close(fig)


def sensitivity(scale, plt):
    """Uniform sweep vs the method across the N tandas and the checks, EnKF-MC, lattice."""
    s = load_summary(scale)
    s = s[s.filt.isin(["enkf-mc", "enkf-mc-bayes"])].copy()
    s.loc[s.filt == "enkf-mc-bayes", "filt"] = "enkf-mc"
    tandas = [t for t in sorted(s.tanda.unique(), key=lambda t: (not t.startswith("N"), t))
              if t.startswith("N")]
    if not tandas:
        return
    fig, axes = plt.subplots(1, len(tandas), figsize=(3.6 * len(tandas), 3.0), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], tandas):
        sub = s[s.tanda == t]
        fx = sub[sub.kind == "fixed"].groupby("radius")["rmse"]
        med, lo, hi = fx.median(), fx.min(), fx.max()
        ax.errorbar(med.index, med.values, yerr=[med.values - lo.values, hi.values - med.values],
                    fmt="o-", color=color_arm("fixed4"), ms=4, lw=1.2, capsize=2, label="uniform radius")
        for arm in method_arms(sub.arm.unique()):
            g = sub[sub.arm == arm]
            ax.axhline(g.rmse.median(), color=color_arm(arm), lw=1.8, label=label_arm(arm))
            ax.axhspan(g.rmse.min(), g.rmse.max(), color=color_arm(arm), alpha=0.1, lw=0)
        N = int(sub.N.iloc[0]); al = float(sub.alpha.iloc[0]); st = int(sub.stride.iloc[0]) if "stride" in sub else 2
        ax.set_title(f"{t}: N={N}, α={al:g}, lattice spacing {st}")
        ax.set_xlabel("uniform radius"); ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("post-burn-in analysis RMSE in q")
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir(scale), "sensitivity.png"), dpi=150)
    plt.close(fig)


def oneobs(scale, plt):
    """The same radii, one observation per component: masked/group/letkf-only vs global."""
    s = load_summary(scale)
    d = s[s.tanda == "oneobs"]
    if d.empty:
        return
    N = int(d.N.iloc[0])
    g = s[(s.tanda == f"N{N}_s{int(d.stride.iloc[0])}") & (s.network == "lattice") & (s.filt.isin(["enkf-mc", "letkf"]))]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, vals, cols = [], [], []
    for filt in ("enkf-mc", "letkf"):
        fx = g[(g.filt == filt) & (g.kind == "fixed")]
        if len(fx):
            b = fx.groupby("radius")["rmse"].median()
            labels.append(f"{FILT_LABEL[filt]}\nbest uniform r={int(b.idxmin())}"); vals.append(b.min()); cols.append(color_arm("fixed2"))
        for arm in method_arms(g[g.filt == filt].arm.unique()):
            labels.append(f"{FILT_LABEL[filt]}\n{label_arm(arm)}"); vals.append(g[(g.filt == filt) & (g.arm == arm)].rmse.median()); cols.append(color_arm(arm))
    for filt in ("enkf-mc-masked", "enkf-mc-group", "letkf-only"):
        for arm in method_arms(d[d.filt == filt].arm.unique()) + rule_arms(d[d.filt == filt].arm.unique()):
            labels.append(f"{FILT_LABEL[filt]}\n{label_arm(arm)}"); vals.append(d[(d.filt == filt) & (d.arm == arm)].rmse.median()); cols.append(color_arm(arm))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("post-burn-in analysis RMSE in q"); ax.set_title(f"global analysis vs one observation per component (lattice, N={N})")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir(scale), "oneobs.png"), dpi=150); plt.close(fig)


def ablation(scale, plt):
    """E3: post-burn-in RMSE of every variant against the reference arms."""
    s = load_summary(scale)
    d = s[s.tanda.str.startswith("E3_")]
    if d.empty:
        return
    d = d.assign(name=d.tanda.str.replace("E3_", "") + ": " + d.arm)
    d = d.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(d) + 1.5))
    cols = ["#b2182b" if a == "bayes" else "#4d4d4d" for a in d.arm]
    ax.barh(d.name, d.rmse, color=cols); ax.set_xlabel("post-burn-in analysis RMSE in q"); ax.invert_yaxis()
    ax.set_title("E3: sensitivity and ablation of the Bayesian rows (random-fixed 10%, one seed)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir(scale), "ablation.png"), dpi=150); plt.close(fig)


def main(scale):
    plt = setup_matplotlib()
    s = load_summary(scale)
    for t in sorted(t for t in s.tanda.unique() if t.startswith("N")):
        rmse_vs_cycle(scale, plt, t)
        rmse_vs_radius(scale, plt, t)
        divergence_map(scale, plt, t)
    sensitivity(scale, plt)
    oneobs(scale, plt)
    ablation(scale, plt)
    print("benchmark figures ->", fig_dir(scale))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCALE", "smoke"))
