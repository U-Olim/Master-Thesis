# High-Dimensional IVQR under Weak Instruments

This repository contains the code for a master thesis Monte Carlo study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The project studies finite-sample performance of instrumental variable quantile regression estimators in high-dimensional designs with weak instruments.

## Estimators

- **Oracle IVQR**: infeasible benchmark using the true active control set.
- **Post-selection IVQR**: feasible sparse-control benchmark using Lasso-based control selection.
- **DML-style residualized IVQR**: cross-fitted residualized IVQR estimator for high-dimensional controls.
- **Full-control IVQR benchmark**: separate benchmark that uses all controls directly.

The full-control benchmark is kept outside the main simulation because it is slow and can be unstable in high-dimensional settings.

## Methodological Caveat

The reported DML estimator is a DML-style residualized IVQR estimator with cross-fitting. It should not be interpreted as an exact density-weighted Chen-Huang-Tien DML-IVQR implementation. The implementation is designed for the one-instrument simulation setting used in this thesis.

Alpha-grid resolution matters for alpha estimates, bias, RMSE, coverage, and confidence-region length. All default simulation modes use a 21-point alpha grid on [-1, 3], giving grid step 0.2. Direct estimator fallback grids use the same range and step when `alphas` is not supplied. CLI options or explicit `alphas` can override the default grid.

## Repository Structure

```text
src/
  config.py                 Compatibility import surface; simulation.config is canonical.
  dgp/                      Data-generating processes and true parameters.
  estimators/               Oracle, post-selection, DML-style, and full-control IVQR estimators.
  inference/                Alpha grids, moments, metrics, and confidence-region utilities.
  reporting/                Summary aggregation, tables, and figures.
  simulation/               Simulation configuration, runner, batching, chunking, validation.
  utils/                    Shared validation helpers.

scenarios/
  main_simulation.py        Main fast/full simulation runner.
  full_control_ivqr.py      Separate full-control IVQR benchmark runner.
  _common.py                Shared scenario-script helpers.

tests/
  config/                   Project-configuration tests.
  dgp/                      DGP and true-parameter tests.
  estimators/               Estimator tests.
  inference/                Inference and metric tests.
  reporting/                Summary/table/figure tests.
  simulation/               Simulation runner, batching, chunking, and config tests.
  utils/                    Shared utility-validation tests.
  test_scripts.py           CLI and end-to-end smoke tests.
```

## Environment Setup

Pixi is the project environment manager.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After cloning the repository, use Pixi from the project root:

```bash
pixi run import_check
pixi run test_project
pixi run test_slow
```

## Simulation Design

Main simulation defaults are shared by fast and full mode except for the number of replications.

```text
DGPs:                 dgp1, dgp2, dgp3
Sample sizes:         n = 500, 1000
Control dimensions:   p = 200, 500
Instrument strengths: pi = 1.0, 0.5, 0.25, 0.10
Quantiles:            tau = 0.25, 0.50, 0.75
Fast replications:    R = 10
Full replications:    R = 500
Alpha grid:           21 points on [-1, 3], step 0.2
Estimators:           oracle, dml, post_selection
DML folds:            K = 3
Parallel jobs:        n_jobs = 4
Batch size:           10 designs
Main base seed:       12345
```

Full-control benchmark defaults:

```text
DGPs:                 dgp1
Sample sizes:         n = 500, 1000
Control dimensions:   p = 20, 50, 100
Instrument strengths: pi = 1.0
Quantiles:            tau = 0.25, 0.5, 0.75
Replications:         R = 500
Alpha grid:           21 points on [-1, 3], step 0.2
Estimator:            full_control_ivqr
Parallel jobs:        n_jobs = 4
Batch size:           10 designs
Base seed:            54321
```

## Main Commands

```bash
pixi run import_check
pixi run test_project
pixi run test_slow
pixi run fast_mode
pixi run full_mode
pixi run full_control
```

Recommended local fast-mode command for a MacBook Pro M5 base model:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

caffeinate -dimsu pixi run python scenarios/main_simulation.py \
  --mode fast \
  --estimators oracle dml post_selection \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --output results/raw/fast_mode_results_grid21.csv \
  --manifest results/raw/fast_mode_manifest_grid21.json
```

Resume the same local run with the same output and manifest pair:

```bash
caffeinate -dimsu pixi run python scenarios/main_simulation.py \
  --mode fast \
  --estimators oracle dml post_selection \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --resume \
  --output results/raw/fast_mode_results_grid21.csv \
  --manifest results/raw/fast_mode_manifest_grid21.json
```

Run a targeted estimator subset when diagnosing runtime or coverage:

```bash
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --estimators oracle \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21

pixi run python scenarios/main_simulation.py \
  --mode fast \
  --estimators dml \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21

pixi run python scenarios/main_simulation.py \
  --mode fast \
  --estimators oracle post_selection \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21
```

Recommended diagnostic workflow: run `--estimators oracle`, then
`--estimators post_selection`, then `--estimators dml`, and reserve the full
estimator set for final comparisons.

## Experiment A: Wider alpha-grid sensitivity

Experiment A tests whether low coverage and high confidence-region boundary-hit
rates are partly caused by truncating confidence regions at the default
`[-1, 3]` alpha-grid boundaries. It keeps the same grid resolution as the main
21-point grid by using 31 points on `[-2, 4]`, so the implied step remains 0.2.
Do not reuse baseline output or manifest files for this run.

Run oracle and post-selection first because they are completed and faster:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-min -2 \
  --alpha-max 4 \
  --alpha-grid-size 31 \
  --estimators oracle post_selection \
  --output results/raw/fast_grid31_wide_oracle_post.csv \
  --manifest results/raw/fast_grid31_wide_oracle_post_manifest.json
```

Resume the same wider-grid run with the same output and manifest pair:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-min -2 \
  --alpha-max 4 \
  --alpha-grid-size 31 \
  --estimators oracle post_selection \
  --resume \
  --output results/raw/fast_grid31_wide_oracle_post.csv \
  --manifest results/raw/fast_grid31_wide_oracle_post_manifest.json
```

DML can be tested later after the oracle/post-selection sensitivity confirms
whether boundary truncation matters. The resume manifest signature includes
`alpha_min`, `alpha_max`, `alpha_grid_size`, and `estimators`, so a 31-point
wide-grid run cannot be resumed with a baseline 21-point manifest.

Compare the baseline 21-grid files with the wider-grid result using the existing
summary aggregation:

```zsh
pixi run python -c "from reporting.summaries import compare_result_files; compare_result_files(['results/raw/fast_grid21_oracle_only.csv', 'results/raw/fast_grid21_post_selection_only.csv', 'results/raw/fast_grid31_wide_oracle_post.csv'], ['grid21_oracle', 'grid21_post_selection', 'grid31_wide_oracle_post'], expected_replications=10).to_csv('results/summary/experiment_a_wide_grid_comparison.csv', index=False)"
```

Review `coverage`, `bias`, `mae`, `rmse`, `avg_cr_length`,
`avg_cr_hull_length`, `cr_boundary_hit_rate`, `alpha_hat_boundary_rate`,
`cr_disconnected_rate`, `mean_failed_alpha_rate`, and runtime columns by design.

## Experiment B: Quantile-specific post-selection

Experiment B adds an opt-in estimator, `post_selection_quantile`, without
changing the baseline `post_selection` estimator. The baseline post-selection
estimator selects controls using mean-regression LassoCV for `Y ~ X` and `D ~ X`.
The quantile-specific variant selects outcome controls using L1-penalized
quantile regression at the design quantile `tau`, selects treatment controls
using the existing mean LassoCV for `D ~ X`, takes the union, retains all
excluded instruments, and then runs the same inverse-IVQR alpha-grid inference
with that fixed selected control set.

This experiment tests whether tau-aligned selection improves coverage and point
estimation. It does not fully solve post-selection inference uncertainty and
should be interpreted as an experimental sensitivity check.

Run the quantile-specific post-selection estimator only:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --estimators post_selection_quantile \
  --output results/raw/fast_grid21_post_selection_quantile_only.csv \
  --manifest results/raw/fast_grid21_post_selection_quantile_only_manifest.json
```

Resume the same run with the same output and manifest pair:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --estimators post_selection_quantile \
  --resume \
  --output results/raw/fast_grid21_post_selection_quantile_only.csv \
  --manifest results/raw/fast_grid21_post_selection_quantile_only_manifest.json
```

For a direct baseline comparison, run:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --estimators post_selection post_selection_quantile \
  --output results/raw/fast_grid21_post_selection_vs_quantile.csv \
  --manifest results/raw/fast_grid21_post_selection_vs_quantile_manifest.json
```

The quantile-selection penalty grid is `(0.001, 0.003, 0.01, 0.03, 0.1)` with
3-fold CV. PSQ-specific diagnostics use the `psq_` prefix, including selected
quantile penalty, selected-control counts, shares, warning code, and stage
runtime columns.

## Experiment C: IVQR-aligned post-selection

Experiment C adds another opt-in estimator, `post_selection_ivqr_aligned`,
without changing baseline `post_selection` or Experiment B
`post_selection_quantile`. Baseline post-selection selects controls from
mean-Lasso `Y ~ X` and `D ~ X`. Experiment B selects outcome controls from
quantile-specific `Y ~ X`. Experiment C selects controls from quantile-L1
regressions of the transformed inverse-IVQR outcome
`Y - alpha_anchor * D` on `X`.

Alpha anchors are data-independent quartiles of the searched alpha interval:
`alpha_min + {0.25, 0.50, 0.75} * (alpha_max - alpha_min)`. Thus the default
`[-1, 3]` grid uses anchors `0, 1, 2`, while the wider `[-2, 4]` grid uses
`-0.5, 1, 2.5`. The selected controls are the union across anchor regressions
plus treatment controls from the same `D ~ X` LassoCV used by baseline
post-selection. This fixed selected set is then held constant during final
alpha-grid inversion. Instruments are retained, not selected.

This experiment tests whether aligning selection with the inverse-IVQR
transformed outcome improves coverage and point estimates. It is still not
fully selection-robust inference; it is an experimental comparator.

Run the IVQR-aligned estimator only:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --estimators post_selection_ivqr_aligned \
  --output results/raw/fast_grid21_post_selection_ivqr_aligned_only.csv \
  --manifest results/raw/fast_grid21_post_selection_ivqr_aligned_only_manifest.json
```

Resume the same run with the same output and manifest pair:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --estimators post_selection_ivqr_aligned \
  --resume \
  --output results/raw/fast_grid21_post_selection_ivqr_aligned_only.csv \
  --manifest results/raw/fast_grid21_post_selection_ivqr_aligned_only_manifest.json
```

The quantile-selection penalty grid is `(0.001, 0.003, 0.01, 0.03, 0.1)` with
3-fold CV. PSA-specific diagnostics use the `psa_` prefix, including anchor
metadata, selected-control counts and shares, selected penalties by anchor,
anchor-failure counts, and stage runtime columns.

## Experiment D: Conservative critical-value sensitivity

Experiment D adds an explicit confidence-region calibration sensitivity through
`--critical-value-multiplier`. The nominal chi-square critical value is still
computed as before, but confidence-region inversion uses:

```text
critical_value_adjusted = critical_value_nominal * critical_value_multiplier
```

The accepted confidence-region grid points are those with
`test_stat(alpha) <= critical_value_adjusted`. The default multiplier is `1.0`,
so the nominal baseline is unchanged unless the option is passed explicitly.
This is a sensitivity experiment, not a replacement for the main inference
procedure.

The multiplier is applied to estimators that use inverse-IVQR confidence-region
inversion: `oracle`, `post_selection`, `post_selection_quantile`,
`post_selection_ivqr_aligned`, `dml`, and the separate `full_control_ivqr`
scenario. It does not change DGPs, estimator formulas, selection rules, DML
nuisance logic, alpha-grid defaults, or the definition of `alpha_hat`.
`alpha_hat` remains the minimizer of the test statistic over the grid.

Result rows include:

```text
critical_value_nominal
critical_value_multiplier
critical_value_adjusted
critical_value
```

For backward compatibility, `critical_value` is the threshold actually used for
the confidence region, so it equals `critical_value_adjusted`. The multiplier is
also part of the manifest resume signature; a run with multiplier `1.0` cannot
resume into a manifest created with multiplier `1.10`, and vice versa.

Baseline nominal oracle/post-selection:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --critical-value-multiplier 1.0 \
  --estimators oracle post_selection \
  --output results/raw/fast_grid21_cv100_oracle_post.csv \
  --manifest results/raw/fast_grid21_cv100_oracle_post_manifest.json
```

Moderate conservative sensitivity:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --critical-value-multiplier 1.10 \
  --estimators oracle post_selection \
  --output results/raw/fast_grid21_cv110_oracle_post.csv \
  --manifest results/raw/fast_grid21_cv110_oracle_post_manifest.json
```

Strong conservative sensitivity:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-grid-size 21 \
  --critical-value-multiplier 1.20 \
  --estimators oracle post_selection \
  --output results/raw/fast_grid21_cv120_oracle_post.csv \
  --manifest results/raw/fast_grid21_cv120_oracle_post_manifest.json
```

Combined with the wider Experiment A grid:

```zsh
pixi run python scenarios/main_simulation.py \
  --mode fast \
  --n-jobs 4 \
  --batch-size 10 \
  --alpha-min -2 \
  --alpha-max 4 \
  --alpha-grid-size 31 \
  --critical-value-multiplier 1.10 \
  --estimators oracle post_selection \
  --output results/raw/fast_grid31_wide_cv110_oracle_post.csv \
  --manifest results/raw/fast_grid31_wide_cv110_oracle_post_manifest.json
```

Use separate output and manifest files for each multiplier. Candidate values
for the sensitivity table are `1.00`, `1.05`, `1.10`, and `1.20`. Point
estimation metrics should remain unchanged across multiplier runs with the same
designs and alpha grid; coverage, CR length, CR hull length, empty-region rate,
disconnected-region rate, and boundary-hit rates may change.

## Dry Runs

Use dry runs to inspect resolved defaults without writing results.

```bash
python scenarios/main_simulation.py --mode fast --dry-run
python scenarios/main_simulation.py --mode full --dry-run
python scenarios/full_control_ivqr.py --dry-run
```

The dry-run output includes `alpha_min`, `alpha_max`, `alpha_grid_size`, and the implied `alpha_grid_step`.

## Output Locations

Default raw outputs:

```text
Fast mode:       results/raw/fast_mode_results.csv
Full mode:       results/raw/full_mode_results.csv
Full-control:    results/raw/full_control_ivqr_results.csv
```

Default summary outputs:

```text
Fast mode:       results/summary/fast_mode_summary.csv
Full mode:       results/summary/full_mode_summary.csv
Full-control:    results/summary/full_control_ivqr_summary.csv
```

Default table directories:

```text
Fast mode:       results/tables/fast
Full mode:       results/tables/full
Full-control:    results/tables/full_control
```

Default figure directories:

```text
Fast mode:       results/figures/fast
Full mode:       results/figures/full
Full-control:    results/figures/full_control
```

No default manifest path is set. A manifest is written only when `--manifest PATH` is supplied.

## Generated Tables and Figures

Standard table files:

```text
comparison_table.csv
diagnostic_table.csv
bias_wide.csv
rmse_wide.csv
mae_wide.csv
coverage_wide.csv
cr_length_wide.csv
cr_hull_length_wide.csv
runtime_wide.csv
failure_rate_wide.csv
```

Boundary diagnostics in the summary and diagnostic tables distinguish point
estimates from confidence-region truncation. `alpha_hat_boundary_rate` is the
share of rows where `alpha_hat` lies at the searched alpha-grid boundary.
`cr_boundary_hit_rate` is the share of rows where the confidence region touches
the searched alpha-grid boundary; high values mean confidence-region endpoints
may be truncated by the searched grid.

Post-selection diagnostics distinguish selected controls from retained
instruments. The current post-selection IVQR estimator selects controls and
retains all excluded instruments, reported as
`ps_instrument_selection_method = all_instruments_retained`. The legacy
`ps_n_selected_instruments` and `ps_share_selected_instruments` fields are kept
as compatibility aliases for retained instruments; use
`ps_n_retained_instruments`, `ps_share_retained_instruments`, and
`ps_all_instruments_retained` for interpretation.

Runtime diagnostics are written to every raw result row. `runtime_seconds` is
kept for compatibility and matches `runtime_total_sec`; estimator-specific
columns such as `dml_runtime_crossfit_sec`, `ps_runtime_selection_sec`,
`oracle_runtime_alpha_loop_sec`, and `fc_runtime_alpha_loop_sec` identify coarse
bottlenecks. Stages that are not cleanly separable without refactoring are left
missing rather than estimated indirectly.

Standard figure files:

```text
fig_bias.png
fig_rmse.png
fig_coverage.png
fig_cr_length.png
fig_failure_rate.png
```

## Resume, Failed Rows, Chunking, and Manifests

Use `--resume` to append to an existing results CSV while skipping design keys that already have the required estimator rows. Resume requires an existing `--manifest PATH` so the current run settings can be checked against the stored resume signature before any output is written. Always use a dedicated output file and manifest file for each configuration. In the main simulation, a design is considered complete when all requested estimator rows are present. In the full-control benchmark, completion is checked for the `full_control_ivqr` estimator.

Use `--rerun-failed` with `--resume` to ignore failed prior rows when deciding whether a design is complete, so failed designs can be rerun. Without `--resume`, `--rerun-failed` has no effect.

Use `--num-chunks N --chunk-index I` to run one deterministic strided chunk of the design grid, with `0 <= I < N`. Both options must be supplied together. Chunking is applied before resume filtering.

Use `--manifest PATH` to write a JSON manifest containing run parameters, counts, alpha-grid information, and a resume signature. When resuming, the stored resume signature must match the current run settings; otherwise the run is rejected.

## Reproducibility

Simulation design seeds are deterministic. The main simulation uses base seed `12345`; the full-control benchmark uses base seed `54321`. The design seed is a deterministic function of DGP, `n`, `p`, `pi`, `tau`, and replication index.

DML fold splits use the design seed by default. Post-selection LassoCV selection randomness also uses the design seed in simulation runs. Parallel batch execution sorts result rows after worker completion, so serial and parallel output ordering is deterministic for the same design set.

Direct estimator calls still expose their own default random-state parameters. The simulation runner ties estimator-level randomness to the design seed.

## Testing

Normal tests exclude slow smoke/integration tests:

```bash
pixi run test_project
```

Slow tests are run separately:

```bash
pixi run test_slow
```

Import checks are available through:

```bash
pixi run import_check
```

## Generated and Ignored Folders

The repository ignores generated outputs and local tooling caches, including:

```text
results/
.pytest_tmp/
.pytest_cache/
.ruff_cache/
.mypy_cache/
__pycache__/
.agents/
.pixi/
```
