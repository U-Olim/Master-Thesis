# High-Dimensional IVQR Monte Carlo Study

This repository contains thesis code for Monte Carlo simulations of instrumental
variable quantile regression under high-dimensional controls and weak
instruments.

The project compares three estimators:

- Oracle IVQR
- Post-selection IVQR (mean-Lasso union followed by CH inverse-IVQR)
- DML-style residualized IVQR

The generic simulation entry point is:

```powershell
pixi run fast
```

## Structure

```text
scenarios/
  run_simulation.py

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
pixi run fast
pixi run full
pixi run import_check
pixi run results
pixi run test
pixi run test_slow
```

`pixi run test` runs the fast suite and deselects tests marked `slow`.
`pixi run test_slow` runs only tests marked `slow`.

## Simulation Modes

- `fast`: `R = 10`
- `full`: `R = 500`

Default estimators are `oracle post_selection dml`.

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

Fast default:

```powershell
pixi run fast
```

Full default:

```powershell
pixi run full
```

Fast all estimators:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

Full default with explicit output:

```powershell
pixi run python scenarios/run_simulation.py --mode full --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/full_results.csv --manifest results/raw/full_manifest.json
```

`pixi run fast` and `pixi run full` retain the generic project defaults. In
particular, their DML defaults are not the tuned settings used for the final
thesis results.

Generic dry run:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --dry-run
```

### Dedicated estimator runners

Each dedicated runner locks execution to one estimator while reusing the same
production grid, seed, DGP, parallel execution, resume, manifest, and serializer
paths as generic single-estimator execution:

```powershell
pixi run python scenarios/run_oracle_ivqr.py --reps 10 --output results/raw/oracle_fast.csv
pixi run python scenarios/run_post_selection_ivqr.py --reps 10 --output results/raw/post_selection_fast.csv
pixi run python scenarios/run_dml_ivqr.py --reps 10 --output results/raw/dml_fast.csv
```

The generic `scenarios/run_simulation.py` runner remains available temporarily.
The dedicated runners are behaviorally equivalent to its single-estimator mode;
generic `full` mode remains available and has not yet been removed.

### Canonical final thesis production commands

The following tasks explicitly encode the final R=500 specification and write
to the canonical raw-result paths consumed by the analysis layer:

```powershell
pixi run final_oracle
pixi run final_post_selection
pixi run final_dml
```

These are expensive production simulations. They use full mode, 500
replications, the 21-point alpha grid on `[-1, 3]`, critical-value multiplier
`1.0`, and base seed `12345`. The post-selection task additionally uses Lasso
multiplier `1.8`; the DML task uses three folds, quantile penalty `0.07`, solver
`highs-ipm`, and ridge alpha `1.0`.

Inspect all three resolved configurations without running simulations:

```powershell
pixi run final_dry_run
```

The canonical tasks express complete replication indices 0--499. Equivalent
jobs may be split with `--rep-start` and `--rep-end`, using distinct block
outputs and manifests before assembling the canonical files. The repository
contains historical block-command examples, but its final provenance manifest
does not establish whether the canonical R=500 files were executed as single
runs or in blocks.

### Historical calibration and robustness commands

Full R500 runs can be split into replication blocks with `--rep-start` and
`--rep-end`. `--reps` remains the total planned replication count, while the
block arguments select the global replication indices for the current process.
Each block should use a separate output and manifest file.

Oracle R500 block 0-99:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle --reps 500 --rep-start 0 --rep-end 99 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/oracle_R500_grid41_block000_099.csv --manifest results/raw/oracle_R500_grid41_block000_099_manifest.json
```

Post-selection R500 block 0-99:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --reps 500 --rep-start 0 --rep-end 99 --selection-lasso-multiplier 1.8 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/post_selection_R500_lasso180_grid41_block000_099.csv --manifest results/raw/post_selection_R500_lasso180_grid41_block000_099_manifest.json
```

Fast DML only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

DML chosen experimental setting:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --reps 5 --dml-k-folds 3 --dml-quantile-penalty 0.07 --dml-quantile-solver highs-ipm --alpha-min -1 --alpha-max 3 --alpha-grid-size 21 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/dml_R5_penalty007_solver_highsipm.csv --manifest results/raw/dml_R5_penalty007_solver_highsipm_manifest.json
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
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --base-seed 12345 --output results/raw/fast_post_selection.csv --manifest results/raw/fast_post_selection_manifest.json
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
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --selection-lasso-multiplier 1.2 --base-seed 12345 --output results/raw/fast_post_selection_lasso120.csv --manifest results/raw/fast_post_selection_lasso120_manifest.json
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
PCs and later merged, as long as they use the same mode, design settings, and
base seed. These runs generate identical data for matching design cells.

Oracle only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle --base-seed 12345 --output results/raw/fast_oracle.csv --manifest results/raw/fast_oracle_manifest.json
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
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --base-seed 12345 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

All estimators:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection dml --base-seed 12345 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

## Output Folders

- `results/raw`: raw estimator-level CSV files and run manifests
- `results/tables`: final thesis tables in CSV and LaTeX formats
- `results/figures`: final thesis figures in PDF and PNG formats

Simulation outputs are ignored by git except for `.gitkeep` placeholders.

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

These three CSVs are the canonical final R=500 inputs. The tracked
`results/raw/manifest.json` records their row and column counts, exact byte
sizes and SHA-256 hashes, and LF-normalized SHA-256 hashes. Exact hashes verify
byte-for-byte equality; canonical hashes allow equivalent LF and CRLF CSV
representations while substantive content changes still fail verification.
Final-run provenance stores the shared grid, seed, and critical-value settings
once under `common`, with estimator-specific tuning and canonical Pixi task
names under `estimators`. The post-selection Lasso multiplier is cross-checked
against its raw result column. The DML fold, penalty, solver, and ridge settings
are documented provenance because those values are not columns in the final DML
raw schema and therefore cannot be verified directly from that CSV.
`pixi run results` verifies the manifest schema, file identities, dimensions,
row totals, and hashes before analysis. Run `pixi run raw_manifest` only when
the canonical raw files intentionally change.

The overview tables summarize broad performance patterns. The canonical
`performance_by_design_cell.csv` and `performance_by_design_cell.tex` tables
preserve results by DGP, sample size, dimensionality, instrument strength,
quantile, and estimator.

## Final Raw Result Schema

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
