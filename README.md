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

Current status: Phase 4 complete.

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

Simulation-runner and metrics aggregation logic will be implemented in later
phases.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Checks

```bash
python -c "import ivqr_sim"
pytest -v
python scripts/01_smoke_test.py
```

## Result Status

`failed=True` means the estimator did not produce a usable estimate.
`converged=True` means the estimator produced `alpha_hat`.
`failed_alpha_count` records failed or sanitized alpha-grid evaluations.
`cr_empty=True` means the inverted confidence region is empty, not that the
estimator failed.
