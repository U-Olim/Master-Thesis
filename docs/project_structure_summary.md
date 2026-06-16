# Project Architecture Summary

This repository uses a `src/` package layout. The main package is `ivqr_sim`.

- `config.py`: fixed thesis design constants.
- `dgp.py`: data-generating processes for DGP1, DGP2, and DGP3.
- `true_effects.py`: true structural quantile treatment effects.
- `moments.py`: IVQR and DML-IVQR moment functions.
- `inference.py`: confidence-region and inference routines.
- `metrics.py`: Monte Carlo performance metrics.
- `estimators/`: Full-control IVQR, Post-selection IVQR, and DML-IVQR.
- `simulation/`: design identifiers, runner orchestration, and aggregation.
- `reporting/`: final thesis tables and figures.
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
- Post-selection IVQR.
- DML-IVQR.
- Weighted GMM objective.
- Pilot simulation runner.
- Full simulation runner with batching and resume support.

Phase 4 estimators use the covariance-weighted GMM objective

```text
n * g_hat(alpha)' Sigma_hat(alpha)^(-1) g_hat(alpha)
```

with a small ridge added to the estimated moment covariance for numerical
stability. The older unweighted quadratic score remains only as a prototype
helper.

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

Quick mode uses `n=100`, `p=20`, `reps=3`, and a 9-point alpha grid. Stress
mode uses `n=250`, `p=200`, `reps=2`, and the same 9-point grid. These pilots
are for checking estimator behavior and runtime, not for final Monte Carlo
conclusions. Estimator iteration limits use realistic pilot defaults rather
than an artificially tiny cap. Full-control IVQR may produce empty confidence
regions or be slow in high-dimensional settings; this is recorded rather than
hidden.

Estimator result status fields use the following convention:

```text
failed=True means the estimator did not produce a usable estimate.
converged=True means the estimator produced alpha_hat.
failed_alpha_count records failed grid evaluations.
cr_empty=True means the inverted confidence region is empty, not that the estimator failed.
```

The full simulation runner follows the binding grid from `Project_structure.pdf`
and writes results batch-by-batch:

```bash
python scripts/03_run_full_simulation.py --resume
python scripts/03_run_full_simulation.py --resume --rerun-failed
python scripts/03_run_full_simulation.py --quick-test --output results/raw/full_quick_test.csv
python scripts/03_run_full_simulation.py --estimators post_selection dml --reps 10 --resume
```

The default run uses 3 DGPs, 3 sample sizes, 3 control dimensions, 4
instrument strengths, 3 quantiles, and 1000 replications, so it can be
computationally expensive. `--resume` skips designs for which all requested
estimator rows already exist in the output CSV. `--rerun-failed` makes resume
stricter: failed estimator rows are not treated as completed. If one estimator
crashes unexpectedly, the runner records a failed row for that estimator and
continues with the remaining estimators for the same dataset. Full-control IVQR
is included by default because infeasibility in high-dimensional scenarios is
informative and is recorded as estimator failure. Diagnostic runs can exclude
it with `--estimators post_selection dml`.

Setup for the `src/` package layout:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -c "import ivqr_sim; print('import ok')"
```

Direct script execution requires the editable install. Without it, the scripts
display a clear installation message instead of a raw `ModuleNotFoundError`.

Basic checks are:

```bash
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
