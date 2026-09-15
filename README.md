# qgloc — per-component localization radii read from the modified-Cholesky precision

Every component is assigned one observed site, and its localization radius is
the distance to that site. The site is chosen from the **partial
correlations** of a *probe* precision: a modified-Cholesky `B⁻¹` built with a
wide uniform radius ρ, never inverted, read entry by entry. The component takes
the observed site it depends on most, given everything else. Those radii fix
the predecessor sets of the precision that the global analysis then uses with
**every** observation. Nothing about the observation values enters the choice:
it is a localization, closed-form per component, with one parameter ρ tied to
the ensemble size.

No search, no metaheuristics. The comparison is against the uniform-radius
sweep in the EnKF-MC and the LETKF, with N ∈ {40, 80, 120} and ρ ∈ {3, 4, 5}.

Two diagnostics are kept on purpose. `oneobs`: the same radii, but each
component updated by its assigned observation only — it loses badly, which is
why assignment chooses radii and not observations. `legacy`: the candidate-based
rules of the first draft (a variational cost `J`, the nearest site, a random
candidate, the raw-variance rule); `J` selected the *least* correlated
candidate, and `paper/nota_criterio.tex` explains why.

## What is in here

```
qgloc/
  testbed.py      QG model (pyteda), climatological ensemble, interior networks
  geometry.py     the grid, its interior, non-periodic square neighbourhoods
  precision.py    modified-Cholesky B^-1, relative ridge, rows cached by (i, r)
  assignment.py   assign_partial (the method); candidates, the cost table
                  and the first-draft rules (diagnostics)
  filters.py      EnKF-MC (precision form) and LETKF, both on the interior,
                  both taking one radius per component; the one-observation
                  updates (masked / group / letkf-only); the cycle
  persist.py      what a run writes so every figure can be rebuilt
experiments/
  common.py               scales (smoke / quick / paper), cached climatology
  exp01_single_cycle.py   the assignment on single forecasts, no analysis
  exp02_benchmark.py      the cycled benchmark: fixed radius vs the rules
figures/
  fig_benchmark.py        RMSE vs cycle, RMSE vs radius, divergence, sensitivity
  fig_assignment.py       radius fields, tessellations, discarded observations,
                          sparsity of B^-1, the fields of a cycle
  make_tables.py          LaTeX tables + FINDINGS-BENCH.md with the numbers
paper/assignment.tex      the paper; \input{tables_paper}
tests/                    fast tests, no model integration
```

## Running it

```
docker compose build
SCALE=smoke docker compose run --rm suite      # ~5 min: checks everything end to end
SCALE=paper docker compose up -d suite         # the real thing (hours; see below)
docker compose logs -f suite
```

`results/` and `paper/` are bind-mounted. `results/cache/` keeps the
climatology (spin-up and snapshots), so a second run starts in seconds.

In parallel, since the benchmark runs are independent:

```
SCALE=paper docker compose run --rm cache          # climatology once
SCALE=paper docker compose up -d shard0 shard1 shard2 shard3   # ~180 h of runs; make shards SHARDS=16 SCALE=paper
SCALE=paper docker compose run --rm figures        # when they are done
```

Or `make paper`, `make shards SHARDS=8 SCALE=paper`, `make figures SCALE=paper`.
Without docker: `make local-test`, `make local-all SCALE=smoke`.

A finished run is resumable: EXP-02 skips any `run.npz` that already exists.

## The experiments

**E1 — first analysis (`exp05_first_analysis.py`)**: one forecast per seed from
the climatology; uniform radii with ridge→0 and with the climatological prior,
the climate lasso structure, LETKF, and the dense climatological covariance as
an oracle; densities 1–40%, N ∈ {20, 40, 80}, ψ and q observed, 20 seeds.

**E2 — cycled (`exp02_benchmark.py`, tandas `N<N>_d<pct>`)**: random networks
(fixed for the run, and redrawn every cycle) observing 25, 10, 5% of the
interior; ψ observed; N ∈ {40, 80}; LETKF r ∈ {1,2,3}, EnKF-MC r ∈ {1,2,3},
EnKF-MC with Bayesian rows; 120 cycles, 60 burn-in, 3 seeds.

**E3 — sensitivity and ablation (tandas `E3_*`)**: taper, α, inflation, w₀,
fixed ρ, no climate, no memory, square-only structure; random-fixed 10%,
N = 40, one seed.

The method: `qgloc/bayes_rows.py` (sequential Bayesian rows: climate prior,
ρ and climate weight by marginal likelihood, tapered ridge) on the structure
`climate lasso ∪ square r=2` (`qgloc/clim_struct.py`).

## What a run leaves on disk

`results/EXP-02_<scale>/<tanda>/<network>/<filter>/<arm>/seed<k>/run.npz`:
per-cycle metrics; observation networks and values; the radius field every
cycle; for assignment arms, the assignment, the observation updating each
component, which observations went unused, the ensemble variance and
innovation at every site, and the full cost table at the snapshot cycles;
background/analysis/truth fields and ensemble spread at the snapshot cycles;
the sparsity of `B^-1` at the snapshot cycles. Plus `run_config.json`.
`figures/*.py` read only these files.

## Decisions worth knowing

- **The boundary is out.** `q` is zero on the Dirichlet boundary. Those
  192 components are not observed, not candidates, not predecessors, and
  not in the error. The state is the 47 × 47 interior.
- **Neighbourhoods do not wrap.** pyteda's are periodic; this domain is not.
  Both filters use `geometry.Grid`, so a radius means the same thing in both.
- **The ridge is relative.** `a_i = α · tr(XᵀX)/p_i`. An absolute constant
  did not regularize (precisions nine orders of magnitude too large, a filter
  that ignored its observations) and would act differently on rows of
  different width, which the assignment produces by design.
- **Climatological ensemble** (Sakov & Oke): members and truth drawn from a
  long run, dissipation ten times the model default so it has a stationary
  regime. Perturb-and-propagate does not work here; `testbed.py` says why.
- **Only `q` is estimated.** `psi` is a diagnostic the model recomputes.
- **The probe radius is tied to N.** Each probe regression has ~ρ² predecessors;
  with N = 40, ρ = 4 works (0.3135 vs 0.320 for the best uniform radius, one
  seed) and ρ = 8 collapses (0.68). ρ is an axis of the benchmark for that
  reason.
- **Divergence is recorded, not avoided.**
