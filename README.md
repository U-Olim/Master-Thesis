# High-Dimensional IVQR under Weak Instruments

This repository contains the code architecture for a master thesis simulation study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The final project compares IVQR-type estimators:

- Full-control IVQR
- Oracle IVQR
- Post-selection IVQR
- DML-IVQR

The planned Monte Carlo design includes:

- DGPs: `dgp1`, `dgp2`, `dgp3`
- Main sample sizes: `n = 500, 1000`
- Main control dimensions: `p = 200, 500`
- Instrument strengths: `pi = 1.0, 0.5, 0.25, 0.10`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Main replications: `R = 100`
- Main alpha grid size: `9`
- Default DML cross-fitting folds: `K = 3`

## Current Status

Current status: Phase 5B full simulation runner implemented.

Implemented:

- Project package architecture
- True structural quantile treatment effects
- `SimData` object
- DGP1, DGP2, and DGP3 data generation
- Corrected nonseparable outcome equation
- Core IVQR moment functions
- Confidence-region inversion
- Full-control IVQR
- Oracle IVQR
- Post-selection IVQR
- DML-IVQR
- Covariance-weighted GMM objective
- Pilot simulation runner
- Full simulation runner with batching and resume support
- Core performance metrics and scenario-level aggregation

The corrected DGP outcome equation is

```text
Y_i = 1 + X_i' beta + D_i(1 + u_i)
```

This implies

```text
alpha_0(tau) = 1 + F_u^{-1}(tau).
```

For DGP1 and DGP2, `F_u` is standard normal. For DGP3, `F_u` is the scaled Student-t distribution used in the heavy-tailed design.

DGP2 uses effective sparsity `s = min(p, 20)` with coefficients

```text
beta_j = 0.5 / sqrt(j)
gamma_j = 0.4 / sqrt(j)
```

for `j = 1, ..., s`, and zero coefficients for `j > s`.

Implemented estimators use the covariance-weighted GMM objective

```text
n * g_hat(a)' * Sigma_hat(a)^(-1) * g_hat(a)
```

with ridge regularization of the estimated moment covariance for numerical
stability.

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

The default quick mode uses `n=100`, `p=20`, `reps=3`, and a 9-point alpha
grid. The stress mode uses `n=250`, `p=200`, `reps=2`, and the same 9-point
grid. These pilots are for checking estimator behavior and runtime, not for
final Monte Carlo conclusions. Estimator iteration limits use realistic pilot
defaults rather than an artificially tiny cap.

Full-control IVQR is benchmark-only. It directly controls for all `X`
variables, so it can be computationally heavy and is not part of the main
high-dimensional default run. It runs when requested explicitly with
`--estimators full` or through the full-control benchmark preset. It raises a
hard error only for invalid inputs or infeasible designs where `p + 1 >= n`,
since the auxiliary quantile regression includes an intercept plus all
controls. No soft warning is emitted merely because `p / n` is high.

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

Oracle IVQR is also simulation-only and infeasible in real applications. It
knows the true active controls from the DGP, restricts `X` to those controls,
and then runs the same IVQR procedure on the reduced control set. It is a
benchmark for post-selection IVQR and DML-IVQR, not an implementable estimator.

The main simulation preset runs Oracle IVQR, post-selection IVQR, and DML-IVQR
on the full high-dimensional grid. It intentionally excludes full-control IVQR,
which is available only through `--preset full-control-benchmark` or an
explicit manual `--estimators full` run.

The default alpha grid size is 9. This is a computationally efficient default,
not a theoretically special value. Larger grids can still be used for
robustness checks with `--alpha-grid-size`, for example
`--alpha-grid-size 13`.

The full simulation runner writes results batch-by-batch and supports resume:

```bash
python scripts/02_run_full_simulation.py --resume
python scripts/02_run_full_simulation.py --resume --rerun-failed
python scripts/02_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
python scripts/02_run_full_simulation.py --estimators oracle post_selection dml --reps 10 --resume
python scripts/02_run_full_simulation.py --preset main
python scripts/02_run_full_simulation.py --preset full-control-benchmark
python scripts/02_run_full_simulation.py --preset main --alpha-grid-size 13
python scripts/02_run_full_simulation.py --preset main --dml-k-folds 5
```

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

The main default grid is computationally expensive: 3 DGPs, 2 sample sizes, 2
control dimensions, 4 instrument strengths, 3 quantiles, and 100 replications,
using a 9-point alpha grid. It runs Oracle IVQR, post-selection IVQR, and
DML-IVQR. Full-control IVQR stays out of the main preset. `--resume` skips
designs for which all requested estimator rows already exist in the output CSV.
`--rerun-failed`
makes resume stricter: failed estimator rows are not treated as completed. If
one estimator raises an exception, the runner records a failed row for that
estimator and continues with the remaining estimators for the same dataset.
Failed rows stay in raw output with `status`, `error_type`, and
`error_message` columns.
DML-IVQR uses 3 cross-fitting folds by default in the runner; use
`--dml-k-folds 5` for a 5-fold robustness run.

Phase 6B aggregation groups raw simulation rows by `dgp`, `n`, `p`, `pi`,
`tau`, and `estimator`. The output contains Monte Carlo metrics and
completeness diagnostics such as observed replications and completion rate.
Performance metrics such as bias, RMSE, coverage, and confidence-region length
are computed on successful rows. Failed rows remain in the raw data and enter
failure-rate diagnostics.
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

Plotting and final figure generation will be implemented in a later phase.

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

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Runnable command scripts live in the top-level `scripts/` folder and can be
run from the repository root.

## Checks

```bash
python -c "import dgp, simulation, estimators, reporting; print('import ok')"
pytest -v
python scripts/01_pilot_simulation.py --mode quick
python scripts/02_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
```

## Result Status

`failed=True` means the estimator did not produce a usable estimate.
`status="ok"` means the row contains a successful estimator result.
`status="failed"` means the row records an estimator exception or failed result.
`error_type` and `error_message` describe failed rows.
`converged=True` means the estimator produced `alpha_hat`.
`failed_alpha_count` records failed or sanitized alpha-grid evaluations.
`cr_empty=True` means the inverted confidence region is empty, not that the
estimator failed.
`cr_length` is the total length of accepted confidence-region blocks. For
disconnected regions, it is not the width between the global lower and upper
accepted alpha values.
