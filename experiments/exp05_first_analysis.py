# -*- coding: utf-8 -*-
"""
EXP-05  The first analysis from a climatological ensemble, exhaustively.

One forecast per seed; every arm analyses the same forecast with the same
observations. Axes: density of the random network, N, observed variable
(psi or q), seeds. Arms: uniform radii with ridge toward zero (best alpha of
a small grid), uniform radii with the climatological prior on the
coefficients (ridge toward beta_clim), the climate lasso structure, the
LETKF, and the dense climatological covariance as an oracle bound.

Writes results/EXP-05_<scale>/first_analysis.csv, one row per (seed, density,
N, obs, arm) with the ratio analysis RMSE / background RMSE.
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ExperimentContext, get_scale, make_testbed, shard_cells
from qgloc.clim_struct import ClimGrid
from qgloc.filters import analysis_letkf, forecast, observe
from qgloc.precision import PrecisionBuilder
from qgloc.progress import Progress, log

EXP_ID = "EXP-05"
ALPHAS0 = (0.1, 0.3, 1.0)          # ridge toward zero
ALPHASC = (1.0, 10.0, 100.0)       # ridge toward beta_clim


def build_binv(grid, DX, struct, alpha, prior):
    nq = grid.n; m = grid.interior
    rows, cols, vals = [], [], []; d = np.empty(nq)
    for i in range(nq):
        rows.append(i); cols.append(i); vals.append(1.0)
        if not m[i]:
            d[i] = 1.0; continue
        idx = struct(i); yv = DX[i]
        if idx.size == 0:
            v = yv.var(); d[i] = 1 / v if v > 0 else 0.0; continue
        X = DX[idx].T; G = X.T @ X; p = idx.size; a = alpha * np.trace(G) / p; G.flat[::p + 1] += a
        rhs = X.T @ yv + (a * prior[i] if prior is not None and i in prior else 0.0)
        beta = np.linalg.solve(G, rhs); v = (yv - X @ beta).var(); d[i] = 1 / v if v > 0 else 0.0
        rows.extend([i] * p); cols.extend(idx.tolist()); vals.extend((-beta).tolist())
    L = csr_matrix((vals, (rows, cols)), shape=(nq, nq))
    return (L.T @ diags(d) @ L).toarray()


def clim_prior(grid, A_c, struct, alpha=0.3):
    out = {}
    for i in np.flatnonzero(grid.interior):
        idx = struct(i)
        if idx.size:
            X = A_c[idx].T; y = A_c[i]; G = X.T @ X; p = idx.size; G.flat[::p + 1] += alpha * np.trace(G) / p
            out[i] = np.linalg.solve(G, X.T @ y)
    return out


def analyse(Qn, Binv, H, y, rng, R):
    N = Qn.shape[1]; A = Binv + (H.T @ H) / R
    D = (y[:, None] + np.sqrt(R) * rng.standard_normal((y.size, N))) - H @ Qn
    return Qn + np.linalg.solve(A, H.T @ D / R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", nargs="?", default=None)
    ap.add_argument("--densities", default="0.01,0.025,0.05,0.10,0.15,0.25,0.40")
    ap.add_argument("--Ns", default="20,40,80")
    ap.add_argument("--obs", default="psi,q")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--network", default="random-fixed")
    a = ap.parse_args()
    scale = get_scale(a.scale)
    ctx = ExperimentContext(EXP_ID, "first analysis from a climatological ensemble", scale)
    dens = [float(x) for x in a.densities.split(",")]; Ns = [int(x) for x in a.Ns.split(",")]
    obs_kinds = a.obs.split(","); seeds = [5000 + 17 * i for i in range(a.seeds)]
    cells = shard_cells([(N, seed) for N in Ns for seed in seeds])
    rows = []
    prog = Progress(len(cells) * len(dens) * len(obs_kinds), label="cases", exp_id=EXP_ID)
    beds = {}
    for N, seed in cells:
        if N not in beds:
            beds[N] = make_testbed(scale, obs_network=a.network, ensemble_size=N, obs_density=0.25)
        bed = beds[N]; grid = bed.grid; m = grid.interior; qb = bed.qblock; nq = bed.nq
        Hpsi = bed.psi_operator()
        clim_q = bed.normalize(bed.snapshots.T.astype(float))[qb, :]; A_c = clim_q - clim_q.mean(1, keepdims=True)
        M = A_c.shape[1]; Bc = A_c @ A_c.T / (M - 1)
        cg = ClimGrid(grid, clim_q, window=6, c=0.5, local=1)
        structs = {f"uniform r={r}": (lambda i, r=r: grid.predecessors(i, r)) for r in (1, 2, 3, 4)}
        structs["climate lasso w6"] = lambda i: cg.pred[i]
        priors = {k: clim_prior(grid, A_c, s) for k, s in structs.items()}
        X0, xt0 = bed.build_ensemble(seed); Xf, xt = forecast(bed, X0, xt0)
        Qn = bed.normalize(Xf)[qb]; qt = bed.normalize(xt)[qb]; DX = Qn - Qn.mean(1, keepdims=True)
        e = lambda Q: float(np.sqrt(np.mean((Q.mean(1) - qt)[m] ** 2))); bg = e(Qn)
        Be = DX @ DX.T / (N - 1)
        binv_cache = {}
        for d_ in dens:
            bed.cfg.obs_density = d_
            obs_idx = bed.network(0, seed=seed)
            for ob in obs_kinds:
                rng = np.random.default_rng(seed * 7 + int(1000 * d_))
                R = bed.cfg.obs_std ** 2
                H = Hpsi[obs_idx] if ob == "psi" else np.eye(nq)[obs_idx]
                y = H @ qt + np.sqrt(R) * rng.standard_normal(obs_idx.size)
                tag = dict(seed=seed, N=N, density=d_, obs=ob, network=a.network, background=bg)
                def rec(arm, Q): rows.append(dict(tag, arm=arm, ratio=e(Q) / bg))
                for k, s in structs.items():
                    for al in ALPHAS0:
                        key = (k, "0", al)
                        if key not in binv_cache: binv_cache[key] = build_binv(grid, DX, s, al, None)
                        rec(f"{k} | ridge->0 | a={al}", analyse(Qn, binv_cache[key], H, y, np.random.default_rng(1), R))
                    for al in ALPHASC:
                        key = (k, "c", al)
                        if key not in binv_cache: binv_cache[key] = build_binv(grid, DX, s, al, priors[k])
                        rec(f"{k} | ridge->clim | a={al}", analyse(Qn, binv_cache[key], H, y, np.random.default_rng(1), R))
                for r in (1, 2, 3):
                    rec(f"LETKF r={r}", analysis_letkf(bed, Qn, np.where(m, r, 0), obs_idx, y, H=(H if ob == "psi" else None)))
                # oracle via Kalman gain (B may be rank deficient)
                S = H @ Bc @ H.T + R * np.eye(obs_idx.size); K = Bc @ H.T @ np.linalg.solve(S, np.eye(obs_idx.size))
                xa = Qn.mean(1) + K @ (y - H @ Qn.mean(1)); rows.append(dict(tag, arm="oracle dense climate B", ratio=float(np.sqrt(np.mean((xa - qt)[m] ** 2))) / bg))
                S = H @ Be @ H.T + R * np.eye(obs_idx.size); K = Be @ H.T @ np.linalg.solve(S, np.eye(obs_idx.size))
                xa = Qn.mean(1) + K @ (y - H @ Qn.mean(1)); rows.append(dict(tag, arm="dense ensemble B (no localization)", ratio=float(np.sqrt(np.mean((xa - qt)[m] ** 2))) / bg))
                prog.step(f"N{N} seed{seed} d={d_} {ob}")
            pd.DataFrame(rows).to_csv(ctx.path("first_analysis_partial.csv"), index=False)
    prog.done()
    df = pd.DataFrame(rows)
    df.to_csv(ctx.path("first_analysis.csv"), index=False)
    best = df[~df.arm.str.contains("LETKF|oracle|dense")].copy()
    best["family"] = best.arm.str.split(" \\| ").str[0] + " | " + best.arm.str.split(" \\| ").str[1]
    g = best.groupby(["obs", "N", "density", "family"]).ratio.mean().groupby(level=[0, 1, 2, 3]).min().unstack("family")
    log("\n" + g.round(3).to_string(), EXP_ID)
    ctx.finish(summary=dict(cases=len(df)))


if __name__ == "__main__":
    main()
