# -*- coding: utf-8 -*-
"""LaTeX tables and the measured-numbers report, from EXP-01 and EXP-02.

Writes ``paper/tables_<scale>.tex`` (all tables, one \\input) and
``results/FINDINGS-BENCH_<scale>.md`` with the numbers stated in prose.

    python figures/make_tables.py paper
"""
from __future__ import annotations

import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fig import (FILT_LABEL, REPO_ROOT, RESULTS_ROOT, exp_dir, label_arm,
                        load_summary, method_arms, rule_arms)


def _arms(arms):
    return sorted([a for a in arms if a.startswith("fixed")], key=lambda a: int(a[5:])) \
        + method_arms(arms) + rule_arms(arms)


def fmt(v, nd=4):
    return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"


def med_range(series, nd=4):
    s = series.dropna()
    if s.empty:
        return "--"
    if len(s) == 1:
        return fmt(float(s.iloc[0]), nd)
    return f"{fmt(s.median(), nd)} [{fmt(s.min(), nd)}, {fmt(s.max(), nd)}]"


def tex_table(body, caption, label, cols):
    return ("\\begin{table}[t]\\centering\\small\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\resizebox{\\linewidth}{!}{%\n"
            f"\\begin{{tabular}}{{{cols}}}\\toprule\n{body}\\bottomrule\\end{{tabular}}}}\\end{{table}}\n\n")


def rmse_table(s, tanda):
    s = s[s.tanda == tanda]
    nets = sorted(s.network.unique()); filts = [f for f in ("enkf-mc", "letkf") if f in set(s.filt)]
    arms = _arms(s.arm.unique())
    head = "arm & " + " & ".join(f"{net} / {FILT_LABEL.get(f, f)}" for net in nets for f in filts) + " \\\\\\midrule\n"
    rows = []
    for a in arms:
        cells = []
        for net in nets:
            for f in filts:
                r = s[(s.network == net) & (s.filt == f) & (s.arm == a)]
                d = int(r.diverged.astype(bool).sum())
                cell = med_range(r.rmse)
                if d:
                    cell += f" ({d}/{len(r)} div.)"
                cells.append(cell)
        name = label_arm(a)
        if a.startswith("partial"):
            name = "\\textbf{" + name + "}"
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body = head + "\n".join(rows) + "\n"
    N = int(s.N.iloc[0]); al = float(s.alpha.iloc[0]); st = int(s.stride.iloc[0])
    geo = f"lattice spacing {st}" if st > 0 else f"random networks observing {100*float(s.density.iloc[0]):.0f}\\% of the interior"
    return tex_table(body, f"Post-burn-in analysis RMSE of $q$ (normalized units), median [min, max] over seeds. "
                     f"$N={N}$, {geo}, $\\alpha={al:g}$. Diverged runs are counted, not averaged.",
                     f"tab:rmse_{tanda}", "l" + "r" * (len(nets) * len(filts)))


def assignment_table(s):
    s = s[s.tanda.str.startswith("N") & (s.kind != "fixed")]
    head = "network / filter & rule & $J$ & mean radius & max radius & not nearest & abstain & obs.\\ unused \\\\\\midrule\n"
    rows = []
    for net in sorted(s.network.unique()):
        for f in sorted(s.filt.unique()):
            for a in method_arms(s.arm.unique()):
                r = s[(s.network == net) & (s.filt == f) & (s.arm == a)]
                if r.empty:
                    continue
                rows.append(f"{net} / {FILT_LABEL.get(f, f)} & {label_arm(a)} (N={int(r.N.min())}--{int(r.N.max())}) & -- & {med_range(r.r_mean, 2)} & "
                            f"-- & {med_range(100*r.frac_not_nearest, 0)}\\% & {med_range(100*r.frac_abstain, 1)}\\% & {med_range(r.obs_unused, 1)} \\\\")
    body = head + "\n".join(rows) + "\n"
    return tex_table(body, "The assignment inside the cycled filter, post-burn-in means, median [min, max] over seeds. "
                     "`not nearest' is the fraction of free components that do not take their nearest observation; "
                     "`abstain' the fraction of interior components updated by no observation.",
                     "tab:assignment", "llrrrrrr")


def timing_table(s):
    s = s[s.tanda == sorted(t for t in s.tanda.unique() if t.startswith("N"))[0]]
    head = "filter & arm & $t_\\mathrm{assign}$ (s) & $t_\\mathrm{analysis}$ (s) \\\\\\midrule\n"
    rows = []
    for f in sorted(s.filt.unique()):
        sub = s[s.filt == f]
        for a in _arms(sub.arm.unique()):
            r = sub[sub.arm == a]
            rows.append(f"{FILT_LABEL.get(f, f)} & {label_arm(a)} & {fmt(r.t_assign.median(), 3)} & {fmt(r.t_analysis.median(), 3)} \\\\")
    return tex_table(head + "\n".join(rows) + "\n", "Wall time per cycle of the assignment and of the analysis, medians over runs and cycles.",
                     "tab:timing", "llrr")


def sensitivity_table(s):
    s = s[(s.network == "lattice") & (s.filt == "enkf-mc")]
    tandas = [t for t in sorted(s.tanda.unique(), key=lambda t: (not t.startswith("N"), t)) if t not in ("oneobs", "legacy")]
    head = "tanda & $N$ & $\\alpha$ & best uniform & RMSE (uniform) & RMSE (method, best $\\rho$) & gain \\\\\\midrule\n"
    rows = []
    for t in tandas:
        sub = s[s.tanda == t]
        fx = sub[sub.kind == "fixed"].groupby("radius")["rmse"].median()
        if fx.empty:
            continue
        rb = int(fx.idxmin())
        pm = sub[sub.kind == "partial"].groupby("arm")["rmse"].median()
        gm = pm.min() if len(pm) else np.nan
        gain = 100 * (1 - gm / fx.min()) if np.isfinite(gm) else np.nan
        rows.append(f"{t} & {int(sub.N.iloc[0])} & {float(sub.alpha.iloc[0]):g} & $r={rb}$ & {fmt(fx.min())} & {fmt(gm)} & {fmt(gain, 1)}\\% \\\\")
    return tex_table(head + "\n".join(rows) + "\n", "Sensitivity to ensemble size and ridge fraction, EnKF-MC on the lattice network. "
                     "Gain is the reduction of the method's RMSE (best probe radius) relative to the best uniform radius.",
                     "tab:sensitivity", "lrrlrrr")


def oneobs_table(s):
    d = s[s.tanda == "oneobs"]
    if d.empty:
        return ""
    N = int(d.N.iloc[0])
    g = s[(s.tanda == f"N{N}") & (s.network == "lattice")]
    head = "update & radii & RMSE \\\\\\midrule\n"
    rows = []
    for f in ("enkf-mc", "letkf"):
        fx = g[(g.filt == f) & (g.kind == "fixed")].groupby("radius")["rmse"].median()
        if len(fx):
            rows.append(f"{FILT_LABEL.get(f, f)}, global, all observations & uniform $r={int(fx.idxmin())}$ (best) & {fmt(fx.min())} \\\\")
        for a in method_arms(g[g.filt == f].arm.unique()):
            rows.append(f"{FILT_LABEL.get(f, f)}, global, all observations & {label_arm(a)} & {med_range(g[(g.filt == f) & (g.arm == a)].rmse)} \\\\")
    for f in ("enkf-mc-masked", "enkf-mc-group", "letkf-only"):
        for a in _arms(d[d.filt == f].arm.unique()):
            rows.append(f"{FILT_LABEL.get(f, f)}, one observation per component & {label_arm(a)} & {med_range(d[(d.filt == f) & (d.arm == a)].rmse)} \\\\")
    return tex_table(head + "\n".join(rows) + "\n", f"The same radii, two updates: the global analysis with every observation, and each component "
                     f"updated by its assigned observation only (lattice, $N={N}$). Assignment chooses radii, not observations.",
                     "tab:oneobs", "llr")


def legacy_table(s):
    d = s[s.tanda == "legacy"]
    if d.empty:
        return ""
    N = int(d.N.iloc[0])
    g = s[(s.tanda == f"N{N}_s{int(d.stride.iloc[0])}") & (s.network == "lattice") & (s.filt == "enkf-mc")]
    head = "rule for the radius (global EnKF-MC) & mean radius & not nearest & RMSE \\\\\\midrule\n"
    rows = []
    fx = g[g.kind == "fixed"].groupby("radius")["rmse"].median()
    if len(fx):
        rows.append(f"uniform $r={int(fx.idxmin())}$ (best of the sweep) & {int(fx.idxmin())} & -- & {fmt(fx.min())} \\\\")
    for a in method_arms(g.arm.unique()):
        r = g[g.arm == a]; rows.append(f"{label_arm(a)} & {med_range(r.r_mean, 2)} & {med_range(100*r.frac_not_nearest, 0)}\\% & {med_range(r.rmse)} \\\\")
    for a in rule_arms(d.arm.unique()):
        r = d[d.arm == a]; rows.append(f"{label_arm(a)} & {med_range(r.r_mean, 2)} & {med_range(100*r.frac_not_nearest, 0)}\\% & {med_range(r.rmse)} \\\\")
    return tex_table(head + "\n".join(rows) + "\n", f"The candidate-based rules of the first draft against the method, all in the global EnKF-MC "
                     f"(lattice, $N={N}$). The rule by $J$ picks the least correlated candidate; the raw-variance rule picks sampling noise at distance.",
                     "tab:legacy", "lrrr")


def single_cycle_table(scale):
    p = os.path.join(exp_dir("EXP-01", scale), "single_cycle.csv")
    if not os.path.exists(p):
        return "", None
    d = pd.read_csv(p)
    parts = sorted({c[7:] for c in d.columns if c.startswith("r_mean_partial")}, key=lambda a: int(a[7:]))
    head = "network & cycle & rule & mean radius & max radius & not nearest & nnz($B^{-1}$) & nnz uniform (same mean) & $t$ probe/table & $t$ $B^{-1}$ \\\\\\midrule\n"
    rows = []
    for (net, st, cyc), sub in d.groupby(["network", "axis", "cycle"]):
        for a in parts:
            rows.append(f"{net} {st} & {int(cyc)} & {label_arm(a)} & {med_range(sub['r_mean_'+a], 2)} & {med_range(sub['r_max_'+a], 0)} & "
                        f"{med_range(100*sub['not_nearest_'+a], 0)}\\% & {med_range(sub['nnz_'+a], 0)} & {med_range(sub.nnz_uniform, 0)} & "
                        f"{fmt(sub['t_probe_'+a].median()*1e3, 0)} ms & {fmt(sub['t_B_'+a].median()*1e3, 0)} ms \\\\")
        for a in ("greedy", "nearest"):
            rows.append(f"{net} {st} & {int(cyc)} & {label_arm(a)} & {med_range(sub['r_mean_'+a], 2)} & {med_range(sub['r_max_'+a], 0)} & "
                        f"{med_range(100*sub['not_nearest_'+a], 0)}\\% & {med_range(sub['nnz_'+a], 0)} & -- & {fmt(sub.t_table.median()*1e3, 0)} ms & {fmt(sub['t_B_'+a].median()*1e3, 0)} ms \\\\")
    return tex_table(head + "\n".join(rows) + "\n", "The radii on single forecasts (EXP-01): the method for each probe radius, and the first-draft rules, "
                     "with the structure of $B^{-1}$ each induces. Medians [min, max] over seeds.",
                     "tab:single_cycle", "llllrrrrrr"), d


def findings_md(scale, s, d1):
    L = [f"# FINDINGS-BENCH ({scale})", "",
         "Every number below is read from `results/EXP-01_*/single_cycle.csv` and `results/EXP-02_*/summary.csv`.", ""]
    if "spread_q_clim" in s.columns:
        sq = float(s.spread_q_clim.iloc[0]); so = float(s.obs_std_raw.iloc[0])
        L += [f"Units: RMSE is in units of the climatological standard deviation of q, {sq:.4g} in model units "
              f"(1.0 = as bad as not assimilating). Observation noise 0.05 = {so:.4g} in model units. "
              f"`summary.csv` also has `rmse_raw` in model units.", ""]
    for t in sorted(t for t in s.tanda.unique() if t.startswith("N")):
        m = s[s.tanda == t]
        geo = f"lattice spacing {int(m.stride.iloc[0])}" if int(m.stride.iloc[0]) > 0 else f"random networks, {100*float(m.density.iloc[0]):.0f}% observed"
        L += [f"## {t}: N = {int(m.N.iloc[0])}, {geo}, alpha = {float(m.alpha.iloc[0]):g}, {m.seed.nunique()} seed(s), {int(m.n_cycles.max())} cycles", ""]
        for net in sorted(m.network.unique()):
            for f in sorted(m.filt.unique()):
                sub = m[(m.network == net) & (m.filt == f)]
                fx = sub[sub.kind == "fixed"].groupby("radius")["rmse"].median()
                if fx.empty:
                    continue
                line = f"- **{net} / {FILT_LABEL.get(f, f)}**: best uniform r={int(fx.idxmin())} at {fx.min():.4f}"
                for a in method_arms(sub.arm.unique()):
                    r = sub[sub.arm == a]; v = r.rmse.median(); dv = int(r.diverged.astype(bool).sum())
                    line += f"; {a} {v:.4f} ({100*(1-v/fx.min()):+.1f}%, mean radius {r.r_mean.median():.2f}, {100*r.frac_not_nearest.median():.0f}% not nearest" + (f", {dv} diverged" if dv else "") + ")"
                L.append(line)
        L.append("")
    for t in ("oneobs", "legacy", "alpha"):
        sub = s[s.tanda == t]
        if sub.empty:
            continue
        L += [f"## {t} (N={int(sub.N.iloc[0])}, alpha={float(sub.alpha.iloc[0]):g}, stride {int(sub.stride.iloc[0]) if 'stride' in sub else '?'})", ""]
        for f in sorted(sub.filt.unique()):
            ff = sub[sub.filt == f]
            fx = ff[ff.kind == "fixed"].groupby("radius")["rmse"].median()
            parts = [f"uniform r={int(fx.idxmin())} {fx.min():.4f}"] if len(fx) else []
            parts += [f"{a} {ff[ff.arm == a].rmse.median():.4f}" for a in method_arms(ff.arm.unique()) + rule_arms(ff.arm.unique())]
            L.append(f"- {FILT_LABEL.get(f, f)}: " + "; ".join(parts))
        L.append("")
    dv = s[s.diverged.astype(bool)]
    L += ["## Divergence", "", "None." if dv.empty else "\n".join(f"- {r.tanda}/{r.network}/{r.filt}/{r.arm}/seed{r.seed}: cycle {int(r.diverged_at)}" for r in dv.itertuples()), ""]
    if d1 is not None:
        parts = sorted({c[7:] for c in d1.columns if c.startswith("r_mean_partial")})
        L += ["## Single forecasts (EXP-01)", ""]
        for (net, st, cyc), sub in d1.groupby(["network", "axis", "cycle"]):
            for a in parts:
                L.append(f"- {net} {st}, cycle {int(cyc)}, {a}: mean radius {sub['r_mean_'+a].median():.2f}, max {sub['r_max_'+a].median():.0f}, "
                         f"{100*sub['not_nearest_'+a].median():.0f}% not nearest, nnz(B^-1) {sub['nnz_'+a].median():.0f} vs uniform {sub.nnz_uniform.median():.0f} "
                         f"(same mean radius {int(sub.r_uniform.median())}); probe {sub['t_probe_'+a].median()*1e3:.0f} ms, B^-1 {sub['t_B_'+a].median()*1e3:.0f} ms.")
            L.append(f"- {net} {st}, cycle {int(cyc)}, first-draft J: greedy J {sub.J_greedy.median():.4f} vs nearest {sub.J_nearest.median():.3f}, "
                     f"{100*sub.not_nearest_greedy.median():.0f}% not nearest, {sub.obs_unused_greedy.median():.0f} obs unused (the rule that chose the least correlated candidate).")
    return "\n".join(L) + "\n"


def main(scale):
    s = load_summary(scale)
    tex = "% generated by figures/make_tables.py — do not edit\n"
    for t in sorted(t for t in s.tanda.unique() if t.startswith("N")):
        tex += rmse_table(s, t)
    tex += assignment_table(s) + timing_table(s) + sensitivity_table(s) + oneobs_table(s) + legacy_table(s)
    sc_tex, d1 = single_cycle_table(scale)
    tex += sc_tex
    out = os.path.join(REPO_ROOT, "paper", f"tables_{scale}.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    md = findings_md(scale, s, d1)
    outm = os.path.join(RESULTS_ROOT, f"FINDINGS-BENCH_{scale}.md")
    with open(outm, "w") as fh:
        fh.write(md)
    print("tables ->", out); print("findings ->", outm)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCALE", "smoke"))
