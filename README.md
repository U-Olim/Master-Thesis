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
runtime_wide.csv
failure_rate_wide.csv
```

Standard figure files:

```text
fig_bias.png
fig_rmse.png
fig_coverage.png
fig_cr_length.png
fig_failure_rate.png
```

## Resume, Failed Rows, Chunking, and Manifests

Use `--resume` to append to an existing results CSV while skipping design keys that already have the required estimator rows. In the main simulation, a design is considered complete when all requested estimator rows are present. In the full-control benchmark, completion is checked for the `full_control_ivqr` estimator.

Use `--rerun-failed` with `--resume` to ignore failed prior rows when deciding whether a design is complete, so failed designs can be rerun. Without `--resume`, `--rerun-failed` has no effect.

Use `--num-chunks N --chunk-index I` to run one deterministic strided chunk of the design grid, with `0 <= I < N`. Both options must be supplied together. Chunking is applied before resume filtering.

Use `--manifest PATH` to write a JSON manifest containing run parameters, counts, alpha-grid information, and a resume signature. When resuming with a manifest, the stored resume signature must match the current run settings; otherwise the run is rejected.

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
