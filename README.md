# High-Dimensional IVQR Monte Carlo Study

This repository contains thesis code for Monte Carlo simulations of instrumental
variable quantile regression under high-dimensional controls and weak
instruments.

The project compares three estimators:

- Oracle IVQR
- Post-selection IVQR (mean-Lasso union followed by CH inverse-IVQR)
- DML-style residualized IVQR

The primary simulation entry points are:

```powershell
pixi run oracle
pixi run post_selection
pixi run dml
```

## Structure

```text
scenarios/
  run_oracle_ivqr.py
  run_post_selection_ivqr.py
  run_dml_ivqr.py
  run_simulation.py        Retired generic-CLI migration stub.

src/
  analysis/     Final-result validation, tables, and figures.
  dgp/          Simulation designs and data-generating processes.
  estimators/   Oracle, post-selection, and DML estimators.
  ivqr/         Alpha grids, CH inverse-IVQR, moments, and confidence regions.
  simulation/   Simulation config, runner, and result-row construction.
  utils/        Validation and timing helpers.

documents/
  Experiments.qmd  Experiment-design and simulation-run documentation.
  Experiments.pdf  Rendered experiment-design document.

scripts/
  make_results.py

tests/          Unit and integration tests for simulation and analysis code.
```

## Environment

Pixi is the only project manager used by this repository.

```powershell
pixi run oracle
pixi run post_selection
pixi run dml
pixi run import_check
pixi run results
pixi run test
pixi run test_slow
```

`pixi run test` runs the fast suite and deselects tests marked `slow`.
`pixi run test_slow` runs only tests marked `slow`.

## Simulation execution

Each supported command runs exactly one estimator. Dedicated runners default to
`R = 10`; production tasks set `R = 500` explicitly.

## Design Grid

| DGP | Role | Active controls |
|---|---|---:|
| `dgp1` | Baseline sparse Gaussian design | 5 |
| `dgp2` | Denser sparse selection-stress design | 10 |
| `dgp3` | Heavy-tail sparse robustness design | 5 |

The reduced sparsity keeps the baseline design cleaner while high dimensionality
is maintained through `p = 200` and `p = 500`. Weak-instrument difficulty is
still controlled by `pi`.

## Simulation Commands

### Development commands

Oracle development run:

```powershell
pixi run oracle
```

Post-selection development run:

```powershell
pixi run post_selection
```

DML development run:

```powershell
pixi run dml
```

### Dedicated estimator runners

Each dedicated runner locks execution to one estimator while reusing the same
production grid, seed, DGP, parallel execution, resume, manifest, and serializer
infrastructure directly:

```powershell
pixi run python scenarios/run_oracle_ivqr.py --reps 10 --output results/experiments/oracle_fast.csv
pixi run python scenarios/run_post_selection_ivqr.py --reps 10 --output results/experiments/post_selection_fast.csv
pixi run python scenarios/run_dml_ivqr.py --reps 10 --output results/experiments/dml_fast.csv
```

The former multi-estimator full mode was removed. Run Oracle, Post-selection and
DML separately through their dedicated entry points. All runners retain the
same deterministic design-seed mapping, so corresponding designs use the same
simulated data. The generic `scenarios/run_simulation.py` path is retired and
now exits with a migration message; it cannot execute simulations.

Resume through the same dedicated estimator runner that created the output:

```powershell
pixi run python scenarios/run_oracle_ivqr.py --resume --output results/experiments/oracle_fast.csv --manifest results/experiments/oracle_fast_manifest.json
```

Dedicated output contracts are fixed at 26 columns for Oracle, 52 columns for
Post-selection, and 43 columns for DML.

### Canonical final thesis production commands

The following tasks explicitly encode the historical R=500 specification for
future reproduction runs. They write to `results/experiments/` and never
overwrite the canonical thesis artifacts:

```powershell
pixi run final_oracle
pixi run final_post_selection
pixi run final_dml
```

These are expensive production simulations. They use 500 replications, the
21-point alpha grid on `[-1, 3]`, critical-value multiplier
`1.0`, and base seed `12345`. The post-selection task additionally uses Lasso
multiplier `1.8`; the DML task uses three folds, quantile penalty `0.07`, solver
`highs-ipm`, and ridge alpha `1.0`.

Inspect all three resolved configurations without running simulations:

```powershell
pixi run final_dry_run
```

These reproduction tasks express complete replication indices 0--499.
Equivalent jobs may be split with `--rep-start` and `--rep-end`, using distinct
block outputs and manifests. The final provenance manifest does not establish
whether the frozen thesis artifacts were executed as single runs or in blocks.

### Historical calibration and robustness commands

Full R500 runs can be split into replication blocks with `--rep-start` and
`--rep-end`. `--reps` remains the total planned replication count, while the
block arguments select the global replication indices for the current process.
Each block should use a separate output and manifest file.

Oracle R500 block 0-99:

```powershell
pixi run python scenarios/run_oracle_ivqr.py --reps 500 --rep-start 0 --rep-end 99 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/blocks/oracle_R500_grid41_block000_099.csv --manifest results/blocks/oracle_R500_grid41_block000_099_manifest.json
```

Post-selection R500 block 0-99:

```powershell
pixi run python scenarios/run_post_selection_ivqr.py --reps 500 --rep-start 0 --rep-end 99 --selection-lasso-multiplier 1.8 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/blocks/post_selection_R500_lasso180_grid41_block000_099.csv --manifest results/blocks/post_selection_R500_lasso180_grid41_block000_099_manifest.json
```

Fast DML only:

```powershell
pixi run python scenarios/run_dml_ivqr.py --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/experiments/fast_dml.csv --manifest results/experiments/fast_dml_manifest.json
```

DML chosen experimental setting:

```powershell
pixi run python scenarios/run_dml_ivqr.py --reps 5 --dml-k-folds 3 --dml-quantile-penalty 0.07 --dml-quantile-solver highs-ipm --alpha-min -1 --alpha-max 3 --alpha-grid-size 21 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/experiments/dml_R5_penalty007_solver_highsipm.csv --manifest results/experiments/dml_R5_penalty007_solver_highsipm_manifest.json
```

For parallel DML batches, cap numerical-library threads to avoid process/thread
oversubscription:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Baseline post-selection:

```powershell
pixi run python scenarios/run_post_selection_ivqr.py --base-seed 12345 --output results/experiments/fast_post_selection.csv --manifest results/experiments/fast_post_selection_manifest.json
```

## Post-selection Lasso Multiplier

`--selection-lasso-multiplier` controls only the Lasso control-selection step
inside post-selection IVQR. The default is `1.0`, which preserves baseline
behavior. It is separate from `--critical-value-multiplier`, which affects
confidence-region inversion.

Values above `1.0` increase the Lasso penalty selected by cross-validation and
therefore tend to select fewer controls. This can reduce over-selection, but
large values can under-select controls and introduce omitted-control bias.

Post-selection with a 1.2 multiplier:

```powershell
pixi run python scenarios/run_post_selection_ivqr.py --selection-lasso-multiplier 1.2 --base-seed 12345 --output results/experiments/fast_post_selection_lasso120.csv --manifest results/experiments/fast_post_selection_lasso120_manifest.json
```

## QuantReg Iteration-Warning Policy

The shared Chernozhukov--Hansen inverse-IVQR API uses
`iteration_warning_policy="use_if_valid"` in normal production operation. A
Statsmodels `IterationLimitWarning` fit is retained only when its parameters,
covariance, excluded-instrument block, and Wald statistic pass all validity
checks; it remains recorded as not converged.

Set `iteration_warning_policy="reject"` explicitly when reproducing legacy
Oracle or post-selection simulation results. That mode rejects every
iteration-limit fit through the historical failed-alpha path. The DML
estimator does not use this policy.

Genuine unusable CH alpha evaluations use `hard_failure_policy="unresolved"`
in production. They are excluded from the usable-only point-estimate argmin,
are neither accepted nor rejected, and cannot be used for confidence-region
boundary interpolation. Confidence-region and coverage outputs explicitly
record partial or full numerical non-resolution. Set
`hard_failure_policy="legacy_reject"` only to reproduce the historical
sentinel-statistic rejection behavior. This policy applies to Oracle and
post-selection CH inference only; DML estimation is unchanged.

## CH Alpha-Grid Strategy

Oracle and post-selection use `grid_strategy="adaptive"` in production. The
midpoint-assisted adaptive boundary refinement first evaluates the configured
initial grid, probes each adjacent usable initial interval once, and then
bisects detected accepted/rejected transitions. Transitions are refined
widest-first with deterministic tie-breaking. Refinement stops at tolerance
`0.025`, depth `10`, or `201` total alpha evaluations; unresolved points are
barriers and are never refined or interpolated through. Evaluations are cached.
This controlled search reduces the risk of missing narrow islands or gaps but
does not guarantee discovery of every disconnected confidence region.

The production point-estimate rule is `alpha_hat_grid="initial"`: boundary
resolution does not redefine the point estimator. Set
`alpha_hat_grid="all_evaluated"` to reproduce the earlier adaptive argmin over
initial, midpoint, and refined points. Fixed mode gives the same point estimate
under either setting because it adds no adaptive points.

Set `grid_strategy="fixed"` (CLI: `--grid-strategy fixed`) to reproduce the
original fixed-grid estimator and its initial-grid point estimate. The adaptive
limits can be configured with `--refinement-tolerance`,
`--max-refinement-depth`, and `--max-alpha-evaluations`. These options apply
only to the shared CH path; DML keeps its supplied fixed alpha grid and does not
receive the CH refinement options.

## Post-selection methodology

The feasible benchmark is **mean-Lasso union selection followed by CH
inverse-IVQR**. It fits ordinary mean LassoCV models for Y on X and D on X,
uses the union of selected controls, retains all instruments, and then applies
CH inversion. Canonical metadata records `selection_method="mean_lasso_union"`,
conditional-mean targets, `selection_quantile_specific=False`,
`instrument_selection_method="retain_all"`, and no post-selection inference
adjustment. It is not quantile-Lasso or alpha-specific selection, is not
cross-fitted or orthogonal-score DML, and does not provide formally
selection-adjusted inference.

`selection_random_state` remains accepted as deprecated compatibility metadata.
The cyclic LassoCV/default-CV configuration is deterministic, so the value does
not affect selection. New reports use **retained instruments**; historical
`ps_n_selected_instruments` fields remain compatibility aliases only.

## Compatibility paths

Historical reproducibility paths intentionally retained are
`iteration_warning_policy="reject"`,
`hard_failure_policy="legacy_reject"`, `grid_strategy="fixed"`,
`adaptive_midpoint_probe=False`, and
`alpha_hat_grid="all_evaluated"`. The fixed-grid path, warning rejection,
legacy hard-failure sentinel, deprecated selected-instrument aliases, and
deprecated `selection_random_state` parsing remain supported. Production
defaults are respectively `"use_if_valid"`, `"unresolved"`, `"adaptive"`,
midpoint probing enabled, and `alpha_hat_grid="initial"`. These CH settings are
inherited by Oracle and post-selection only; DML does not call or accept them.

CH result rows preserve the complete confidence region in `cr_components` as
compact JSON, for example `[[-1.0,-0.42],[0.18,1.36]]`. Internally the same
geometry is an immutable `tuple[tuple[float, float], ...]`. `cr_lower` and
`cr_upper` are only the outer hull endpoints, while `cr_length` is the sum of
component lengths; a disconnected hull must not be read as one accepted
interval. Empty or fully unresolved CH regions serialize as `[]` and remain
distinguishable through `cr_status`. DML uses a null component value. Readers
retain older rows without this column as “components unavailable” and never
reconstruct components from hull bounds.

## Reproducibility and Separate Estimator Runs

The project uses a fixed default base seed, `12345`. Each design cell has a
deterministic seed derived from `base_seed`, `dgp`, `n`, `p`, `pi`, `tau`, and
`rep`.

The design seed does not depend on estimator name, estimator order, number of
workers, or batch size. Therefore, estimators can be run separately on different
PCs and later merged, as long as they use the same design settings and
base seed. These runs generate identical data for matching design cells.

Oracle only:

```powershell
pixi run python scenarios/run_oracle_ivqr.py --base-seed 12345 --output results/experiments/fast_oracle.csv --manifest results/experiments/fast_oracle_manifest.json
```

Oracle-only simulation CSVs intentionally contain exactly these 26 fields, in
this order:

```text
dgp,n,p,pi,tau,rep,alpha_true,alpha_hat,covered,cr_length,cr_status,cr_n_blocks,cr_disconnected,cr_components,iteration_warning_evaluations,seed,cr_lower,cr_upper,converged,cr_is_numerically_resolved,cr_unresolved_count,final_alpha_evaluations,refinement_depth_reached,number_of_refined_intervals,minimum_final_grid_spacing,median_final_grid_spacing
```

The output retains exact confidence-region geometry, direct interval endpoints,
the per-observation seed, convergence and resolution indicators, warning counts,
and concise adaptive-grid effort and spacing diagnostics. Constant run settings
and lower-level midpoint, limit, and refinement-barrier diagnostics remain in the
command manifest or internal estimator diagnostics. Resume and block merging
accept historical expanded Oracle files, validate all retained fields, and write
this schema. Post-selection and DML CSV schemas are unchanged.

DML only:

```powershell
pixi run python scenarios/run_dml_ivqr.py --base-seed 12345 --output results/experiments/fast_dml.csv --manifest results/experiments/fast_dml_manifest.json
```

Post-selection only:

```powershell
pixi run python scenarios/run_post_selection_ivqr.py --base-seed 12345 --output results/experiments/fast_post_selection.csv --manifest results/experiments/fast_post_selection_manifest.json
```

## Output Folders

- `results/raw`: immutable, validated thesis artifacts and their provenance manifest
- `results/blocks`: temporary replication-block outputs
- `results/experiments`: new simulation and reproduction outputs
- `results/tables`: final thesis tables in CSV and LaTeX formats
- `results/figures`: final thesis figures in PDF and PNG formats

The files under `results/raw/` are frozen thesis artifacts. New simulations
should be written to `results/blocks/`, `results/experiments/`, or another
explicit output path. Simulation outputs are ignored by git except for
`.gitkeep` placeholders.

## Release Packaging

A clean repository archive should exclude local environments, caches, and test
artifacts: `.pixi/`, `__pycache__/`, `.pytest_tmp/`, `.pytest_cache/`, and
`.ruff_cache/`. These paths are already protected by `.gitignore`; packaging a
release does not require deleting the working environment.

## Final Results Analysis

Simulation and final analysis are separate workflows. After the completed R=500
files are present under `results/raw`, generate the thesis outputs with:

```powershell
pixi run results
```

This command loads and validates the oracle, post-selection, and DML result
files, then writes the final tables to `results/tables/` and figures to
`results/figures/`. It does not run simulations or alter the raw files.

The canonical final R=500 inputs are exactly:

- `results/raw/oracle_ivqr.csv`
- `results/raw/post_selection_ivqr.csv`
- `results/raw/dml_ivqr.csv`

Analysis reads these immutable files in place; it never copies, projects, or
rewrites them on disk. The tracked `results/raw/manifest.json` records exact
paths, ordered columns, dimensions, replication coverage, natural-key checks,
byte sizes, and SHA-256 hashes. It also states that `pre-refactor-r500` is the
known validation reference, not a claim about the unknown creating commit.
`pixi run results` verifies this metadata before analysis. The raw-manifest task
is historical-maintenance tooling and must only be run when intentionally
recording the existing canonical artifacts.

The overview tables summarize broad performance patterns. The canonical
`performance_by_design_cell.csv` and `performance_by_design_cell.tex` tables
preserve results by DGP, sample size, dimensionality, instrument strength,
quantile, and estimator.

## Historical and Current Result Schemas

The validated historical thesis artifacts retain the schemas with which the
R=500 study was completed:

- Oracle: 43 columns
- Post-selection: 52 columns
- DML: 15 columns

Current serializers for future simulations emit different estimator-specific
contracts:

- Oracle: 26 columns
- Post-selection: 52 columns
- DML: 43 columns

Serializer evolution does not invalidate or trigger rewriting of historical
artifacts. Analysis validates only the historical fields required for the
requested tables and figures, and harmonizes compatible fields in memory.

All three completed R=500 result files contain:

- design identifiers: `dgp`, `n`, `p`, `pi`, `tau`, `rep`, and `seed`;
- the estimator label `estimator`;
- the point estimate `alpha_hat` and true parameter `alpha_true`;
- confidence-region hull bounds `cr_lower` and `cr_upper`, together with the
  reported accepted-set length `cr_length`;
- the coverage indicator `covered` and convergence indicator `converged`.

Grid-inverted confidence regions may be disconnected. Consequently,
`cr_length` may be smaller than `cr_upper - cr_lower`, and coverage cannot
always be inferred from hull inclusion alone. The post-selection file also
contains `n_selected_controls` and `selection_lasso_multiplier`.

Aggregate metrics such as bias, MAE, RMSE, and coverage probability are
computed by the final analysis workflow rather than stored as raw columns.

## Notes

The DML estimator is a DML-style residualized IVQR implementation. It uses
cross-fitting and residualized nuisance components, but it should not be read as
an exact density-weighted Chen-Huang-Tien DML-IVQR implementation.
