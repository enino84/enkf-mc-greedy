# -*- coding: utf-8 -*-
"""
What one run leaves on disk, so that every figure can be rebuilt from it.

One ``run.npz`` per (tanda, network, filter, arm, seed), compressed, with:

- ``metrics``: one row per cycle (structured array): errors, spread, radius
  statistics, timings, assignment statistics, divergence flag.
- ``snap_cycles`` and, for each of them, ``xb_q``, ``xa_q``, ``xt_q``,
  ``xb_psi``, ``xa_psi``, ``xt_psi`` (float32, full grid): background mean,
  analysis mean and truth.
- For every cycle: ``obs_idx``, ``y`` (observation network and values),
  ``radius`` (the radius field, int16).
- For assignment arms, every cycle: ``assignment`` (candidate index or
  abstain), ``assigned_obs`` (observation index per component or -1),
  ``obs_used`` (bool per observation), ``obs_var_ens`` (ensemble variance at
  each observation site), ``innovation``, ``J``, and ``cost_table`` (float32)
  at the snapshot cycles only.
- For EnKF-MC-group, at the snapshot cycles: ``domain_<k>_obs``, and per
  observation its group ``members`` and its ``domain`` (group + halo).
- For EnKF-MC-masked, at the snapshot cycles: ``bcol_<k>`` (up to three
  columns of ``B``, i.e. the reach of one observation's correction) and the
  observations they belong to.
- ``binv_cycles`` and, for each, the sparsity of ``B^{-1}`` as
  ``binv_<k>_indptr/indices`` (values are not kept: the structure is what the
  figures show, and it is what the assignment decides).
- ``config.json`` alongside, with everything needed to rerun.

Per-cycle arrays are stacked along axis 0 in cycle order; a run that diverged
at cycle ``k`` has ``k`` entries and ``diverged_at = k``.
"""
from __future__ import annotations

import json
import os

import numpy as np

METRIC_KEYS = ("cycle", "diverged", "p_obs", "spread", "t_assign", "t_analysis",
               "r_mean", "r_max", "b_rmse_q", "b_rmse_psi", "rmse_q", "rmse_psi", "rmse_q_raw", "b_rmse_q_raw",
               "rel_q", "b_rel_q", "rmse", "b_rmse", "J", "frac_not_nearest",
               "frac_abstain", "obs_unused")


class RunRecorder:
    def __init__(self, snap_cycles=(), binv_cycles=(), keep_table_at=()):
        self.snap_cycles = set(int(c) for c in snap_cycles)
        self.binv_cycles = set(int(c) for c in binv_cycles)
        self.keep_table_at = set(int(c) for c in keep_table_at)
        self.rows = []
        self.per_cycle = dict(obs_idx=[], y=[], radius=[])
        self.assign = dict(assignment=[], assigned_obs=[], obs_used=[],
                           obs_var_ens=[], innovation=[])
        self.snaps = {}
        self.tables = {}
        self.binv = {}
        self.domains = {}
        self.bcols = {}
        self.diverged_at = -1

    def cycle(self, k, rec, xb, xa, x_true, Xa, obs_idx, y, rf, aout, Binv,
              extra=None, bed=None):
        self.rows.append(rec)
        extra = extra or {}
        if k in self.snap_cycles:
            if "domains" in extra:
                self.domains[k] = {j: (m.astype(np.int32), D.astype(np.int32))
                                   for j, (m, D) in extra["domains"].items()}
            if "b_columns" in extra:
                self.bcols[k] = extra["b_columns"]
        self.per_cycle["obs_idx"].append(np.asarray(obs_idx, dtype=np.int32))
        self.per_cycle["y"].append(np.asarray(y, dtype=np.float32))
        self.per_cycle["radius"].append(np.asarray(rf, dtype=np.int16))
        if aout is not None:
            self.assign["assignment"].append(np.asarray(aout["assignment"], dtype=np.int8))
            self.assign["assigned_obs"].append(np.asarray(aout["assigned_obs"], dtype=np.int32))
            self.assign["obs_used"].append(np.asarray(aout["obs_used"], dtype=bool))
            self.assign["obs_var_ens"].append(np.asarray(aout["obs_var_ens"], dtype=np.float32))
            self.assign["innovation"].append(np.asarray(aout["innovation"], dtype=np.float32))
            if k in self.keep_table_at:
                self.tables[k] = np.asarray(aout["table"], dtype=np.float32)
        if k in self.snap_cycles:
            self.snaps[k] = dict(xb=np.asarray(xb, np.float32),
                                 xa=np.asarray(xa, np.float32),
                                 xt=np.asarray(x_true, np.float32),
                                 spread=np.asarray(Xa.std(axis=1), np.float32))
        if Binv is not None and k in self.binv_cycles:
            B = Binv.tocsr()
            self.binv[k] = (B.indptr.astype(np.int32), B.indices.astype(np.int32))

    def diverged(self, k):
        self.diverged_at = int(k)

    # ------------------------------------------------------------------
    def write(self, path, config, bed):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        qb, pb_ = bed.qblock, bed.blocks["psi"]
        out = {}
        # metrics as a structured array
        keys = [k for k in METRIC_KEYS if any(k in r for r in self.rows)]
        dt = [(k, "i4" if k in ("cycle", "p_obs", "r_max", "obs_unused") else
               ("?" if k == "diverged" else "f8")) for k in keys]
        m = np.zeros(len(self.rows), dtype=dt)
        for j, r in enumerate(self.rows):
            for k in keys:
                m[k][j] = r.get(k, np.nan if m[k].dtype.kind == "f" else 0)
        out["metrics"] = m
        out["diverged_at"] = np.array(self.diverged_at)
        for k, v in self.per_cycle.items():
            if v:
                out[k] = np.stack(v) if len({a.shape for a in v}) == 1 else \
                    np.array(v, dtype=object)
        for k, v in self.assign.items():
            if v:
                out[k] = np.stack(v) if len({a.shape for a in v}) == 1 else \
                    np.array(v, dtype=object)
        sc = sorted(self.snaps)
        out["snap_cycles"] = np.array(sc, dtype=np.int32)
        if sc:
            for name in ("xb", "xa", "xt", "spread"):
                out[f"{name}_q"] = np.stack([self.snaps[c][name][qb] for c in sc])
                if name != "spread":
                    out[f"{name}_psi"] = np.stack([self.snaps[c][name][pb_] for c in sc])
        out["table_cycles"] = np.array(sorted(self.tables), dtype=np.int32)
        for c, t in self.tables.items():
            out[f"cost_table_{c}"] = t
        out["domain_cycles"] = np.array(sorted(self.domains), dtype=np.int32)
        for c, dd in self.domains.items():
            js = np.array(sorted(dd), dtype=np.int32)
            out[f"domain_{c}_obs"] = js
            out[f"domain_{c}_members"] = np.array([dd[int(j)][0] for j in js], dtype=object)
            out[f"domain_{c}_domain"] = np.array([dd[int(j)][1] for j in js], dtype=object)
        out["bcol_cycles"] = np.array(sorted(self.bcols), dtype=np.int32)
        for c, cc in self.bcols.items():
            out[f"bcol_{c}_obs"] = np.array(sorted(cc), dtype=np.int32)
            if cc:
                out[f"bcol_{c}"] = np.stack([cc[j] for j in sorted(cc)])
        out["binv_cycles"] = np.array(sorted(self.binv), dtype=np.int32)
        for c, (ip, ix) in self.binv.items():
            out[f"binv_{c}_indptr"] = ip
            out[f"binv_{c}_indices"] = ix
        np.savez_compressed(path, **{k: v for k, v in out.items()})
        with open(os.path.splitext(path)[0] + "_config.json", "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        return path


def load_run(path):
    """The ``run.npz`` as a dict, object arrays unpacked."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def metrics_frame(path, **tags):
    """The per-cycle metrics of one run as a DataFrame with the tags attached."""
    import pandas as pd
    z = np.load(path, allow_pickle=True)
    df = pd.DataFrame(z["metrics"])
    for k, v in tags.items():
        df[k] = v
    df["diverged_at"] = int(z["diverged_at"])
    return df
