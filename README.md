# High-Dimensional IVQR under Weak Instruments

This repository contains the code architecture for a master thesis simulation study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The final project will compare three IVQR-type estimators:

- Full-control IVQR
- Post-selection IVQR
- DML-IVQR

The planned Monte Carlo design includes:

- DGPs: `dgp1`, `dgp2`, `dgp3`
- Sample sizes: `n = 250, 500, 1000`
- Control dimensions: `p = 200, 300, 500`
- Instrument strengths: `pi = 1.0, 0.5, 0.25, 0.10`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 1000`
- DML cross-fitting folds: `K = 5`

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
- Post-selection IVQR
- DML-IVQR
- Covariance-weighted GMM objective
- Pilot simulation runner
- Full simulation runner with batching and resume support

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
`QuantileRegressor`, and Ridge residualization of `Z` on `X`. This is a
practical implementation aligned with the score structure in
`Project_structure.pdf`.

The default confidence-region critical value uses `df=1` for scalar-alpha
score inversion. It is not presented as a full overidentification J-test.

The pilot simulation has two diagnostic modes:

```bash
python scripts/02_pilot_simulation.py --mode quick
python scripts/02_pilot_simulation.py --mode stress
python scripts/02_pilot_simulation.py --mode quick --estimators post_selection dml
```

The default quick mode uses `n=100`, `p=20`, `reps=3`, and a 9-point alpha
grid. The stress mode uses `n=250`, `p=200`, `reps=2`, and the same 9-point
grid. These pilots are for checking estimator behavior and runtime, not for
final Monte Carlo conclusions. Estimator iteration limits use realistic pilot
defaults rather than an artificially tiny cap. Full-control IVQR may produce
empty confidence regions or be slow in high-dimensional settings; this is
recorded rather than hidden.

The full simulation runner writes results batch-by-batch and supports resume:

```bash
python scripts/03_run_full_simulation.py --resume
python scripts/03_run_full_simulation.py --resume --rerun-failed
python scripts/03_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
python scripts/03_run_full_simulation.py --estimators post_selection dml --reps 10 --resume
```

The default full grid follows `Project_structure.pdf` and can be
computationally expensive: 3 DGPs, 3 sample sizes, 3 control dimensions, 4
instrument strengths, 3 quantiles, and 1000 replications. `--resume` skips
designs for which all requested estimator rows already exist in the output
CSV. `--rerun-failed` makes resume stricter: failed estimator rows are not
treated as completed. If one estimator crashes unexpectedly, the runner records
a failed row for that estimator and continues with the remaining estimators for
the same dataset. Full-control IVQR is included by default because failure in
high-dimensional settings is informative. To exclude it for diagnostic runs,
use `--estimators post_selection dml`.

Final metrics aggregation will be implemented in a later phase.

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Direct script execution requires the editable install. Without it, the scripts
display a clear installation message instead of a raw `ModuleNotFoundError`.

## Checks

```bash
python -c "import ivqr_sim; print('import ok')"
pytest -v
python scripts/01_smoke_test.py
python scripts/02_pilot_simulation.py --mode quick
python scripts/03_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
```

After installation, equivalent console commands are available:

```bash
ivqr-smoke-test
ivqr-pilot --mode quick
ivqr-full-simulation --quick-test --output results/raw/full_quick_test.csv
```

## Result Status

`failed=True` means the estimator did not produce a usable estimate.
`converged=True` means the estimator produced `alpha_hat`.
`failed_alpha_count` records failed or sanitized alpha-grid evaluations.
`cr_empty=True` means the inverted confidence region is empty, not that the
estimator failed.
