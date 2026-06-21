# Project Architecture Summary

This repository uses a thematic `src/` layout. Core modules and packages live
directly under `src/`.

- `dgp/`: DGP designs, generators, and true structural parameters.
- `estimators/`: Full-control, oracle, post-selection, and DML IVQR estimators.
- `inference/`: alpha grids, moments, confidence regions, and metrics.
- `simulation/`: fixed simulation config, runner orchestration, batching, and chunking.
- `reporting/`: summaries, final thesis tables, and figures.
- `tests/`: verification tests for implemented phases.

## Current Status

Current status: Phase 5B full simulation runner implemented.

Implemented:

- Project package architecture.
- True structural quantile treatment effects.
- `SimData` object.
- DGP1, DGP2, and DGP3 data generation.
- Corrected nonseparable outcome equation.
- Core IVQR moments.
- Confidence-region inversion.
- Full-control IVQR.
- Oracle IVQR.
- Post-selection IVQR.
- DML-IVQR.
- Weighted GMM objective.
- Pilot simulation runner.
- Full simulation runner with batching and resume support.
- Core performance metrics and scenario-level aggregation.

Phase 4 estimators use the covariance-weighted GMM objective

```text
n * g_hat(alpha)' Sigma_hat(alpha)^(-1) g_hat(alpha)
```

with a small ridge added to the estimated moment covariance for numerical
stability. The older unweighted quadratic score remains only as a prototype
helper.

DML-IVQR uses cross-fitting, penalized quantile regression via sklearn
`QuantileRegressor`, and Ridge residualization of `Z` on `X`. The project
default is `dml_k_folds = 3` for computational efficiency in diagnostics and
main simulations. This is not theoretically special; `K=5` remains available
for robustness checks with `--dml-k-folds 5`. This is a practical
implementation aligned with the score structure in `Project_structure.pdf`.

The default confidence-region critical value uses `df=1` for scalar-alpha
score inversion. It is not presented as a full overidentification J-test.

The pilot simulation has two diagnostic modes:

```bash
python scripts/01_pilot_simulation.py --mode quick
python scripts/01_pilot_simulation.py --mode stress
python scripts/01_pilot_simulation.py --mode quick --estimators post_selection dml
```

Quick mode uses `n=100`, `p=20`, `reps=3`, and a 9-point alpha grid. Stress
mode uses `n=250`, `p=200`, `reps=2`, and the same 9-point grid. These pilots
are for checking estimator behavior and runtime, not for final Monte Carlo
conclusions. Estimator iteration limits use realistic pilot defaults rather
than an artificially tiny cap.

Full-control IVQR is benchmark-only. It directly controls for all `X`
variables, so it can be computationally heavy and is not part of the main
high-dimensional default run. It runs when requested manually with
`--estimators full` or via `--preset full-control-benchmark`. It raises a hard
error only for invalid inputs or infeasible designs where `p + 1 >= n`; no
warning is emitted merely because `p / n` is high.

The final full-control benchmark preset is intentionally smaller than the main
high-dimensional simulation:

- Estimator: `full`
- DGPs: `dgp1`, `dgp2`, `dgp3`
- Sample sizes: `n = 500, 1000`
- Control dimensions: `p = 100, 200`
- Instrument strengths: `pi = 1.0, 0.5, 0.25`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 100`
- Alpha grid size: `9`
- Output: `results/raw/full_control_benchmark_R100.csv`

The benchmark excludes `p=500` and `pi=0.10` to keep the full-control run
feasible and focused. Users can still manually run full-control on other
feasible designs with `--estimators full`.

Oracle IVQR is simulation-only and infeasible in real applications. It knows
the true active controls from the DGP, restricts `X` to those controls, and then
runs the same IVQR procedure on the reduced control set. It is a benchmark for
post-selection IVQR and DML-IVQR, not an implementable estimator.

The main simulation preset runs Oracle IVQR, post-selection IVQR, and DML-IVQR
on the full high-dimensional grid with `R = 100` and a 9-point alpha grid. It
intentionally excludes full-control IVQR, which stays in
`--preset full-control-benchmark` or explicit manual `--estimators full` runs.
The 9-point alpha grid is a computationally efficient default, not a
theoretically special value. Larger grids such as 13 points can still be used
as robustness checks with `--alpha-grid-size 13`.

Estimator result status fields use the following convention:

```text
failed=True means the estimator did not produce a usable estimate.
converged=True means the estimator produced alpha_hat.
failed_alpha_count records failed grid evaluations.
cr_empty=True means the inverted confidence region is empty, not that the estimator failed.
cr_length is the total length of accepted confidence-region blocks, not the convex-hull width for disconnected regions.
```

The full simulation runner follows the binding grid from `Project_structure.pdf`
and writes results batch-by-batch:

```bash
python scripts/02_run_full_simulation.py --resume
python scripts/02_run_full_simulation.py --resume --rerun-failed
python scripts/02_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
python scripts/02_run_full_simulation.py --estimators oracle post_selection dml --reps 10 --resume
python scripts/02_run_full_simulation.py --preset main
python scripts/02_run_full_simulation.py --preset full-control-benchmark
python scripts/02_run_full_simulation.py --preset main --alpha-grid-size 13
python scripts/02_run_full_simulation.py --preset main --dml-k-folds 5
python scripts/02_run_full_simulation.py --preset main --reps 10 --n-jobs 6 --output results/raw/main_diagnostic_R10.csv
python scripts/02_run_full_simulation.py --preset main --reps 1 --n-jobs 1 --dry-run
```

`--n-jobs` controls parallel worker processes across independent simulation
designs. The project default is 6 workers; use `--n-jobs 1` for serial/debug
runs. On a laptop, 2-6 workers is the recommended range. Prefer one command
with `--n-jobs` instead of running multiple PowerShell windows against the same
output file, since only one main process should write a given CSV.

Statsmodels `QuantReg` in full-control, oracle, and post-selection IVQR can
occasionally reach its iteration limit during diagnostic runs. The project
default is `--quantreg-max-iter 1000`; increase it when debugging difficult
designs:

```bash
python scripts/02_run_full_simulation.py --preset main --quantreg-max-iter 2000
```

Repeated QuantReg iteration-limit warnings are suppressed in long simulation
logs by default. Use `--show-quantreg-warnings` when you need to inspect those
warnings directly.

Safe final-run planning and chunking examples:

```bash
python scripts/02_run_full_simulation.py --dry-run

python scripts/02_run_full_simulation.py \
  --resume \
  --num-chunks 4 \
  --chunk-index 0 \
  --manifest results/raw/full_chunk_0_manifest.json \
  --output results/raw/full_simulation_results.csv

python scripts/02_run_full_simulation.py \
  --dgps dgp1 \
  --n-values 500 \
  --p-values 200 \
  --pi-values 1.0 0.5 0.25 0.10 \
  --taus 0.5 \
  --reps 50 \
  --estimators oracle post_selection dml \
  --output results/raw/mini_weak_iv.csv

python scripts/02_run_full_simulation.py \
  --preset full-control-benchmark \
  --output results/raw/full_control_benchmark_R100.csv

python scripts/02_run_full_simulation.py \
  --preset main \
  --reps 10 \
  --dml-k-folds 3 \
  --dry-run

python scripts/02_run_full_simulation.py \
  --estimators oracle post_selection dml \
  --reps 10 \
  --n-values 500 \
  --p-values 200 \
  --alpha-grid-size 9 \
  --output results/raw/oracle_post_dml_test.csv
```

The main default run uses 3 DGPs, 2 sample sizes, 2 control dimensions, 4
instrument strengths, 3 quantiles, and 100 replications with a 9-point alpha
grid, so it can be computationally expensive. It runs Oracle IVQR,
post-selection IVQR, and DML-IVQR. Full-control IVQR stays out of the main
preset. `--resume` skips designs for which all requested estimator rows already
exist in the output CSV. `--rerun-failed` makes resume stricter: failed
estimator rows are
not treated as completed. If one estimator raises an exception, the runner
records a failed row for that estimator and continues with the remaining
estimators for the same dataset. Failed rows stay in raw output with `status`,
`error_type`, and `error_message` columns.
DML-IVQR uses 3 cross-fitting folds by default in the runner; use
`--dml-k-folds 5` for a 5-fold robustness run.

Phase 6B aggregation groups raw simulation rows by `dgp`, `n`, `p`, `pi`,
`tau`, and `estimator`. The output contains Monte Carlo metrics and
completeness diagnostics such as observed replications and completion rate.
Performance metrics such as bias, RMSE, coverage, and confidence-region length
are computed on successful rows. Failed rows remain in the raw data and enter
failure-rate diagnostics. The underlying `cr_length` is the total accepted
block length, so disconnected confidence regions do not count rejected gaps as
accepted length.
Aggregation rejects duplicate raw rows for the same `dgp`, `n`, `p`, `pi`,
`tau`, `rep`, `seed`, `estimator` key.

```python
from reporting.summaries import aggregate_results_file

summary = aggregate_results_file(
    "results/raw/full_simulation_results.csv",
    "results/processed/summary_metrics.csv",
    expected_replications=100,
)
```

Phase 6C creates thesis-ready CSV tables for Quarto or manual inclusion:

```bash
python scripts/03_make_tables.py \
  --input results/processed/summary_metrics.csv \
  --output-dir results/tables
```

Raw results can also be aggregated and tabled in one command:

```bash
python scripts/03_make_tables.py \
  --raw-input results/raw/full_simulation_results.csv \
  --summary-output results/processed/summary_metrics.csv \
  --expected-replications 100 \
  --output-dir results/tables
```

Generated files include `comparison_table.csv`, `diagnostic_table.csv`,
`bias_wide.csv`, `rmse_wide.csv`, `mae_wide.csv`, `coverage_wide.csv`,
`cr_length_wide.csv`, `runtime_wide.csv`, and `failure_rate_wide.csv`.

Setup for the `src/` package layout:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -c "import dgp, simulation, estimators, reporting; print('import ok')"
```

Runnable command scripts live in the top-level `scripts/` folder and can be
run from the repository root.

Basic checks are:

```bash
pytest -v
python scripts/01_pilot_simulation.py --mode quick
python scripts/02_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
```

The corrected DGP outcome equation is

```text
Y_i = 1 + X_i' beta + D_i(1 + u_i).
```

This implies

```text
alpha_0(tau) = 1 + F_u^{-1}(tau).
```

For DGP1 and DGP2, `F_u` is standard normal. For DGP3, `F_u` is the scaled Student-t distribution used in the heavy-tailed design.

DGP2 keeps the DGP1 structure but increases the number of relevant controls to
`s = min(p, 20)`. Its nonzero coefficients are

```text
beta_j = 0.5 / sqrt(j)
gamma_j = 0.4 / sqrt(j)
```

for `j = 1, ..., s`; all remaining coefficients are zero.
