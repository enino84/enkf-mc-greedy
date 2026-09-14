# EXPERIMENTS

Two experiments. Both read the same cached climatology; both write everything
they measure. Scales: `smoke` (minutes, checks the pipeline), `quick`
(≈1 h, a first look), `paper` (what the paper reports).

## EXP-01 — the assignment on single forecasts

No analysis is performed. For every seed and both networks, at cycle 0 (a
climatological draw) and after `burn_in` cycles of the EnKF-MC with the method's radii (an
ensemble the filter actually produces):

- the method's radius field for each probe radius ρ, its tessellation, the
  structure of the B⁻¹ it induces, and the cost of the probe
- the first-draft rules (J, nearest, random, variance): their J, radii, and
  which candidates they chose — the record of what J did
- the uniform field with the method's mean radius
- the tessellation (which observation updates each component)
- which observations update nothing; ensemble variance at every site
- sparsity of B⁻¹ under each radius field
- cost of J against cost of assembling B⁻¹

Output: `results/EXP-01_<scale>/<network>/seed<k>/cycle<c>.npz`, `single_cycle.csv`.

## EXP-02 — the cycled benchmark

| tanda | N | α | stride | filters | arms | networks |
|---|---|---|---|---|---|---|
| N<N>_s<stride> | 40 / 80 / 120 | 0.3 | 2 / 3 / 4 | EnKF-MC, LETKF | uniform r ∈ {1,2,3,4,6,8}; partial ρ ∈ {3,4,5} | lattice |
| N<N>_d<pct> | 40 / 80 / 120 | 0.3 | 50% / 25% / 10% | EnKF-MC, LETKF | the same | random-fixed, random-moving |
| oneobs | 40 | 0.3 | 2 | EnKF-MC-masked, EnKF-MC-group, LETKF-only | partial ρ=4; nearest | lattice |
| legacy | 40 | 0.3 | 2 | EnKF-MC | J-greedy, nearest, random, raw-variance | lattice |
| alpha | 40 | 0.1 | 2 | EnKF-MC | r ∈ {1,2,4}; partial ρ=4 | lattice |

Paper scale: 60 cycles, 15 burn-in, 3 seeds. 9 arms × 2 filters × 3 seeds × 3 N × (3 spacings × 1 lattice + 3 densities
× 2 random networks) = 1458 runs in the main tandas, plus 42 in the
diagnostics. Propagation dominates: ≈4 s/cycle at N = 40, ≈8 at 80, ≈12 at
120; ≈180 h on one core. Use shards: 16 shards ≈ 12 h. To shorten, cut
`rhos` to (4,) or `runs` to 2 in `experiments/common.py`.

Reported: post-burn-in analysis RMSE of q over the interior (median and range
over seeds), background RMSE, spread, mean/max radius, fraction of components
not on their nearest site, unused sites, wall time per cycle (probe + analysis),
divergence (cycle at which the forecast failed).

Output: one `run.npz` + `run_config.json` per run (see README), `metrics.csv`
(every cycle of every run), `summary.csv` (one row per run).

## Figures (`results/figures_<scale>/`)

| file | what |
|---|---|
| `rmse_vs_cycle_N<k>` | error curves per arm, median with seed band, per network × filter |
| `rmse_vs_radius_N<k>` | post-burn-in RMSE vs uniform radius; the method (per ρ) as horizontal bands |
| `divergence_N<k>` | fraction of seeds diverged per arm and cell |
| `sensitivity` | uniform sweep vs the method across every N × stride tanda and alpha |
| `oneobs` | global analysis vs one observation per component, same radii |
| `radius_fields_<net>_<filt>` | greedy, nearest, random and uniform radius fields, one colormap; unused obs marked |
| `radius_evolution_<net>_<filt>` | the greedy field along the run |
| `tessellation_<net>_<filt>` | which observation updates each point, nearest vs greedy, and where they differ |
| `discarded_obs_<net>_<filt>` | ensemble spread with unused observations marked; variance histogram used vs unused |
| `sparsity_<net>` | spy(B⁻¹) for the four fields |
| `fields_<net>_<filt>` | truth / background / analysis / error per arm at a snapshot cycle |
| `single_cycle_costs` | J per rule and radius histograms (EXP-01) |

## Tables (`paper/tables_<scale>.tex`)

RMSE per arm and cell; assignment statistics; timing; sensitivity; single
forecasts. `results/FINDINGS-BENCH_<scale>.md` states the same numbers in
prose.
