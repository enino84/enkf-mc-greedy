# -*- coding: utf-8 -*-
"""Shared infrastructure: scales, the cached climatology, output layout, sharding."""
from __future__ import annotations

import json, os, platform, sys, time, warnings
from dataclasses import asdict, dataclass

import numpy as np
warnings.filterwarnings("ignore")

import qgloc
from qgloc.progress import banner, env_summary, fmt_eta, log
from qgloc.testbed import QGConfig, Testbed, build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
CACHE_ROOT = os.path.join(RESULTS_ROOT, "cache")

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = max(1, int(os.environ.get("SHARD_COUNT", "1")))


def shard_cells(cells):
    if SHARD_COUNT == 1:
        return cells
    return [c for i, c in enumerate(cells) if i % SHARD_COUNT == SHARD_INDEX]


@dataclass
class Scale:
    """One size of the suite. ``paper`` is what the paper reports."""
    name: str
    mrefin: int = 5
    spinup: float = 20000.0
    n_snapshots: int = 200
    snapshot_every: float = 250.0
    obs_freq: float = 20.0
    # Stride is an axis: 2 observes 24% of the interior, 3 about 10%, 4
    # about 5.5%. At N = 40 the filter converges at 2, barely at 3 and not
    # at 4 (measured); whether N = 80 or 120 changes that is part of what
    # the benchmark answers.
    obs_stride: int = 2
    strides: tuple = (2, 3, 4)               # lattice spacings
    densities: tuple = (0.5, 0.25, 0.1)      # fraction observed, random networks
    cycles: int = 60
    burn_in: int = 15
    runs: int = 3
    # --- the main benchmark -------------------------------------------
    # Baselines: uniform radius in both filters. Method: radii from the
    # partial correlations of a probe precision of radius rho, in both
    # filters. N is an axis; rho is an axis, because it is tied to N.
    radii: tuple = (1, 2, 3, 4, 6, 8)
    rhos: tuple = (3, 4, 5)
    ensemble_sizes: tuple = (40, 80, 120)
    networks: tuple = ("lattice", "random-fixed", "random-moving")
    ridge_alpha: float = 0.3
    # --- diagnostic tandas, lattice, N = 40 ---------------------------
    # oneobs: the same partial radii, but each component updated by its
    #   assigned observation only (masked / group / letkf-only) -- shows
    #   that assignment chooses radii, not observations.
    # legacy: the candidate-based rules of the first draft (J, nearest,
    #   random, variance) in the global EnKF-MC -- the record of what J did.
    # alpha: sensitivity of the main comparison to the ridge.
    diag_N: int = 40
    diag_rho: int = 4
    diag_radii: tuple = (1, 2, 4)
    legacy_rules: tuple = ("greedy", "nearest", "random", "variance")
    check_alpha: float = 0.1
    snapshot_cycles: tuple = (0, 15, 30, 45, 59)
    kappa: int = 4
    r_max: int = 12

    @property
    def seeds(self):
        return [5000 + 17 * i for i in range(self.runs)]

    def tandas(self):
        """(name, config overrides, arms, filters, networks)."""
        fixed = [("fixed", r) for r in self.radii]
        parts = [("partial", r) for r in self.rhos]
        rand = tuple(n for n in self.networks if n != "lattice")
        for N in self.ensemble_sizes:
            if "lattice" in self.networks:
                for st in self.strides:
                    yield (f"N{N}_s{st}", dict(ensemble_size=N, ridge_alpha=self.ridge_alpha,
                                               obs_stride=st),
                           fixed + parts, ("enkf-mc", "letkf"), ("lattice",))
            for dens in self.densities:
                yield (f"N{N}_d{int(round(100*dens))}",
                       dict(ensemble_size=N, ridge_alpha=self.ridge_alpha, obs_density=dens),
                       fixed + parts, ("enkf-mc", "letkf"), rand)
        dfix = [("fixed", r) for r in self.diag_radii]
        base = dict(ensemble_size=self.diag_N, ridge_alpha=self.ridge_alpha)
        yield ("oneobs", base, [("partial", self.diag_rho), ("nearest", None)],
               ("enkf-mc-masked", "enkf-mc-group", "letkf-only"), ("lattice",))
        yield ("legacy", base, [(r, None) for r in self.legacy_rules], ("enkf-mc",), ("lattice",))
        yield ("alpha", dict(ensemble_size=self.diag_N, ridge_alpha=self.check_alpha),
               dfix + [("partial", self.diag_rho)], ("enkf-mc",), ("lattice",))


SCALES = {
    "smoke": Scale(name="smoke", mrefin=5, spinup=2000.0, n_snapshots=30,
                   snapshot_every=100.0, cycles=4, burn_in=1, runs=1,
                   radii=(1, 2), rhos=(2,), ensemble_sizes=(12,), diag_N=12,
                   diag_rho=2, diag_radii=(1,), legacy_rules=("greedy", "nearest"),
                   obs_stride=3, strides=(3,), densities=(0.1,), snapshot_cycles=(0, 3)),
    "quick": Scale(name="quick", mrefin=5, spinup=8000.0, n_snapshots=80,
                   snapshot_every=200.0, cycles=20, burn_in=5, runs=2,
                   radii=(1, 2, 4), rhos=(3, 4), ensemble_sizes=(40,), strides=(2, 3),
                   densities=(0.25,), snapshot_cycles=(0, 10, 19)),
    "paper": Scale(name="paper"),
}


def get_scale(name=None):
    name = name or os.environ.get("SCALE", "smoke")
    if name not in SCALES:
        raise ValueError(f"unknown scale '{name}'. Available: {sorted(SCALES)}")
    return SCALES[name]


def parse_cli(argv=None):
    import argparse
    global SHARD_INDEX, SHARD_COUNT
    ap = argparse.ArgumentParser(description="Run one qgloc experiment.")
    ap.add_argument("scale", nargs="?", default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--tanda", default=None, help="N<N>_s<stride>, N<N>_d<pct>, oneobs, legacy, alpha; default all")
    a = ap.parse_args(argv)
    if a.shard is not None:
        SHARD_INDEX = a.shard
    if a.shards is not None:
        SHARD_COUNT = max(1, a.shards)
    return a


def config_for(scale, **over):
    cfg = QGConfig(mrefin=scale.mrefin, spinup=scale.spinup,
                   n_snapshots=scale.n_snapshots,
                   snapshot_every=scale.snapshot_every,
                   ensemble_size=scale.diag_N, cycles=scale.cycles,
                   burn_in=scale.burn_in, obs_freq=scale.obs_freq,
                   obs_stride=scale.obs_stride, ridge_alpha=scale.ridge_alpha,
                   kappa=scale.kappa, r_max=scale.r_max, rho=scale.diag_rho)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def climatology(cfg):
    """The long run and its snapshots, cached under results/cache."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    path = os.path.join(
        CACHE_ROOT,
        f"clim_m{cfg.mrefin}_t{int(cfg.spinup)}_n{cfg.n_snapshots}"
        f"_e{int(cfg.snapshot_every)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = build_model(cfg)
    n = model.get_number_of_variables()
    t0 = time.time()
    x = model.propagate(np.zeros(n), np.array([0.0, float(cfg.spinup)]))
    log(f"spin-up to t={cfg.spinup:.0f} in {time.time()-t0:.0f}s")
    snaps = [x.astype(np.float32)]
    for k in range(cfg.n_snapshots):
        x = model.propagate(x, np.array([0.0, float(cfg.snapshot_every)]))
        snaps.append(x.astype(np.float32))
    S = np.array(snaps)
    np.save(path, S)
    log(f"{len(S)} snapshots in {time.time()-t0:.0f}s -> {os.path.basename(path)}")
    return S


def make_testbed(scale, **over):
    cfg = config_for(scale, **over)
    return Testbed(cfg, climatology(cfg))


class ExperimentContext:
    def __init__(self, exp_id, description, scale):
        self.exp_id, self.description, self.scale = exp_id, description, scale
        self.dir = os.path.join(RESULTS_ROOT, f"{exp_id}_{scale.name}")
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time()
        banner(f"{exp_id}   [scale={scale.name}]", [description, f"output: {self.dir}"])
        env_summary()

    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def save_json(self, obj, name):
        p = self.path(name)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        log(f"wrote {name}", self.exp_id)
        return p

    def finish(self, summary=None):
        import pyteda
        man = dict(exp_id=self.exp_id, description=self.description,
                   scale=asdict(self.scale),
                   shard=dict(index=SHARD_INDEX, count=SHARD_COUNT),
                   elapsed_s=round(time.time() - self.t0, 2),
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   versions=dict(qgloc=qgloc.__version__,
                                 pyteda=getattr(pyteda, "__version__", "unknown"),
                                 numpy=np.__version__,
                                 python=sys.version.split()[0],
                                 platform=platform.platform()))
        if summary is not None:
            man["summary"] = summary
        suffix = "" if SHARD_COUNT == 1 else f"_shard{SHARD_INDEX}"
        with open(self.path(f"manifest{suffix}.json"), "w") as fh:
            json.dump(man, fh, indent=2, default=str)
        log(f"DONE in {fmt_eta(man['elapsed_s'])}", self.exp_id)
