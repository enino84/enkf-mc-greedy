# Adaptive localization by observation assignment: findings

> **Status (14 Sep, evening).** The cost `J` below chose the *least*
> correlated candidate: its only dependence on the component is a term
> proportional to σ_ij², minimised at zero correlation. Finding 1 is correct
> as a number and wrong as an interpretation, and findings 5 and 6 follow
> from the same mechanism. The method now reads radii from the partial
> correlations of a probe precision (`assign_partial`); `J` and the other
> candidate rules are kept as the `legacy` diagnostic. `paper/nota_criterio.tex`
> has the derivation and the numbers. What follows is the original record.
>
> These are the measurements that motivated the method, taken on
> single forecasts with the full 49×49 state (boundary included) and an
> absolute ridge of 0.01. Two things have changed since and are reflected in
> the code: the boundary is now excluded from the assimilation entirely
> (`geometry.py`), and the ridge is relative, `alpha·tr(XᵀX)/p`, at α = 0.3
> (`precision.py`) — an absolute ridge did not regularize the cycled filter.
> Neither change touches the results in sections 1–4 and 6, which do not pass
> through the analysis; the boundary change removes the 15 boundary
> observations from section 5 (they no longer exist), and section 7's timings
> are superseded by EXP-02. The benchmark numbers are in
> `results/FINDINGS-BENCH_<scale>.md`, written by `figures/make_tables.py`.


Handoff note. Everything below was measured on the quasi-geostrophic testbed
described at the end; nothing here is an estimate or an expectation.

---

## The idea

Replace the localization radius by an **assignment**. Each model component
grows a neighbourhood until it has collected $\kappa$ candidate observations,
and is then updated by **exactly one** of them, or by none. The radius is not
chosen: it is the distance to whichever observation the component ended up
using.

The cost of an assignment is the variational cost evaluated at the scalar local
analysis. For component $i$ assigned to observation $j$:

$$
x^{a}_{i} = x^{b}_{i} + \frac{\sigma_{ij}}{\sigma_{jj} + r_{j}}\big(y_{j} - x^{b}_{j}\big),
\qquad
J_i = \frac{(x^{a}_{i} - x^{b}_{i})^{2}}{\sigma_{ii}} + \frac{(y_{j} - x^{a}_{j})^{2}}{r_{j}}
$$

and $J_i = (y_j - x^b_j)^2 / r_j$ when the component abstains. The global score
is the mean, $J = \frac{1}{n}\sum_i J_i$.

No truth, nothing withheld, everything from ensemble anomalies. The assignment
then fixes the predecessor sets of a modified-Cholesky precision, assembled
once per cycle for a single global analysis.

---

## Findings

### 1. The nearest observation is usually the wrong one

**79% of components do not use their nearest observation.** With $\kappa = 4$
candidates ordered by distance, the choices split almost evenly:

| candidate | 1st (nearest) | 2nd | 3rd | 4th |
|---|---|---|---|---|
| components | 516 | 603 | 647 | 635 |

What decides is the ensemble correlation, not the distance. This is the central
result: a distance-based rule is choosing badly four times out of five.

### 2. The cost separates good assignments from bad ones

| assignment | $J$ |
|---|---|
| every component abstains | 123.82 |
| every component takes its nearest | 0.587 |
| **optimal assignment** | **0.052** |
| random assignment | 30.28 |

The optimal assignment improves on the nearest-observation rule by a factor of
**eleven**.

### 3. The radius adapts to observation density with no parameter

| network | mean radius | max radius |
|---|---|---|
| regular lattice, spacing 4 | 2.75 | 4 |
| random, same overall density | 1.71 | 5 |

The random network gives a *smaller* mean and a *larger* maximum, because it
leaves both dense clusters and empty regions. Nothing controls this; it falls
out of growing the neighbourhood until $\kappa$ observations are found.

### 4. Same radius budget, different structure

Compared at equal mean radius (3.01 for the assignment, 3.00 for a uniform
radius of 3), the precision matrices differ:

| rule | mean radius | max radius | non-zero in $B^{-1}$ |
|---|---|---|---|
| nearest observation | 1.47 | 3 | 1.15% |
| uniform radius 3 | 3.00 | 3 | 3.07% |
| **optimal assignment** | **3.01** | **7** | **5.53%** |

The assignment does not spend more; it spends differently, concentrating long
radii where they help and short ones where they do not. `sparsity_pattern.png`
and `sparsity_zoom.png` show the rows varying in width.

### 5. The method discards observations where the ensemble has collapsed

Of 144 observations, **97 are used and 47 end up updating nothing**. Two checks
matter here:

**They were not excluded by construction.** All 47 were offered as candidates,
to between 39 and 101 components each (median 68), against a median of 68 for
all observations. They competed and lost everywhere.

**Part of it is a boundary artefact, part is real.** Of the 23 observations
sitting on the Dirichlet boundary, 15 are discarded — trivially, since $q$
vanishes there and the ensemble has no variance. Restricting to the 100
observations at distance $\ge 4$ from the boundary, **23 are still discarded**,
and the discriminant is stark:

| | used | discarded |
|---|---|---|
| ensemble variance at the site (median) | 0.0214 | 0.00058 |
| \|innovation\| (median) | 0.0385 | 0.0633 |
| mean correlation with its candidates | 0.476 | 0.556 |
| offered to N components | 69 | 68 |

A factor of **37 in the ensemble variance**, and nothing else discriminates:
the discarded observations have *larger* innovations and *higher* correlations,
and were offered just as often. The method rejects observations sitting where
the ensemble believes it already knows the answer — where the Kalman gain is
near zero and the observation could not move the analysis anyway. It finds this
with no rule telling it to.

### 6. The optimum is exact

$J$ is a mean of terms each depending only on its own component's choice, so
the optimum is obtained component by component, in closed form: one pass over
a table of $n(\kappa+1)$ precomputed costs. Nothing is searched.

The four rules compared throughout are:

| rule | what it does |
|---|---|
| **greedy** | the optimum of $J$, exact by separability (the method) |
| nearest | every component takes its nearest observation |
| random | a candidate drawn at random per component |
| uniform | a fixed radius everywhere: the standard scheme, enters as a radius field only |

### 7. Cost

| operation | time |
|---|---|
| evaluate $J$ for a whole assignment | 0.016 ms |
| one evaluation requiring the full precision and solve | 180 ms |
| the same, at a radius that destroys sparsity ($r = 12$) | 1070 ms |

Four orders of magnitude. The candidate sets and covariances are precomputed
once per cycle; the search only reads them.

---

## Figures

| file | what it shows |
|---|---|
| `structure_greedy.png` | the three precisions with their values, and the radius field over the domain |
| `sparsity_pattern.png` | binary sparsity pattern, 300 components |
| `sparsity_zoom.png` | the same at 80 components, where rows are visible individually |
| `spatial_assignment.png` | which candidate each point chose; the resulting cells; the same under the nearest-observation rule |
| `why_discarded.png` | where the discarded interior observations sit, and the variance that distinguishes them |

The comparison worth putting in the paper is the second and third panel of
`spatial_assignment.png`: the tessellation the method produces against the
geometric one. They do not look alike.

---

## Testbed

1.5-layer quasi-geostrophic model of Sakov and Oke (2008):

$$
q_t = -\psi_x - \varepsilon J(\psi,q) - A\triangle^{3}\psi + 2\pi\sin(2\pi y),
\qquad q = \triangle\psi - F\psi
$$

on the unit square, Dirichlet boundaries, $F = 1600$, $\varepsilon = 10^{-5}$,
RK4 at $dt = 0.3125$, $49\times49$ per field.

Four settings are consequences of measurements rather than choices, and each
cost real time to find:

**Dissipation ten times the package default.** At the default the norm of $q$
grows by a factor of fourteen between $t = 500$ and $t = 8000$ and never
saturates, so there is no stationary regime and no climatology to sample. At
$10^{-11}$ it settles: $1.107\times10^{4}$ at $t = 20000$ against
$1.222\times10^{4}$ at $t = 60000$. Sakov and Oke report the same.

**Only $q$ is estimated.** The model integrates $q$ and recovers $\psi$ from it
at every step, so an analysis correcting $\psi$ is discarded at the next
propagation. Verified directly: halving $\psi$ and leaving $q$ alone gives a
bit-identical propagation. The state is $q$ alone, $n = 2401$, and $\psi$ is
rebuilt from the analysed $q$.

**Climatological ensemble.** Members and truth drawn from snapshots of one long
run. Perturbing a state and propagating does not work: a perturbation is mostly
energy off the attractor and the hyperviscosity removes it. From 30% of
climatology the background error falls to 0.068 after 120 time units, a factor
of five, and raising the amplitude does not compensate. Only after ~250 units
does the surviving component grow, reaching 0.872 at $t = 1000$ — which is what
a climatological ensemble has from the start.

**$dt = 0.3125$, not 1.25.** At the larger step the forecast blows up when the
network is sparse and the radius short: the analysis corrects observed points
hard and leaves their neighbours alone, and the biharmonic term amplifies the
resulting gradients. The failure is in `propagate`, not in the analysis.
Configurations that still diverge are recorded rather than avoided — which
combinations of density and radius a scheme cannot survive is itself a result.

---

## Code

`qgloc/assignment.py` holds the whole method:

- `build_candidates(g, obs_rows, obs_cols, n_candidates, r_max)` — grows the
  neighbourhoods, returns candidates and their distances
- `LocalCost(DX, cand, obs_idx, y, xb, obs_var)` — precomputes the cost table;
  `.score(assignment)`, `.best_greedy()`, `.radius_field()`, `.assigned_obs()`
- `RULES` — `greedy`, `nearest`, `random`; `uniform_radius(cost, r)` for the
  fixed-radius baseline; `evaluate(cost, rule)` returns assignment, $J$ and
  radius field

`qgloc/precision.py` builds the modified-Cholesky precision from a radius
field, with rows cached by `(component, radius)`. It reproduces pyteda's
estimator; one open discrepancy after excluding boundary components is
documented there and marked `xfail` in the tests.

---

## Open

- Filter performance over cycles with the assignment in the loop has not been
  run end to end, for any of the four rules. Everything above is measured on a
  single forecast.
- The discarded-observation result is from one cycle and one ensemble; it needs
  repeating across cycles and seeds before it goes in a paper as stated.
