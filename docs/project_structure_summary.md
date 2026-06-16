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
- `tests/`: Phase 1/Phase 2 verification tests.

## Current Status

Phase 2 is complete.

Implemented:

- Project package architecture.
- True structural quantile treatment effects.
- `SimData` object.
- DGP1, DGP2, and DGP3 data generation.
- Corrected nonseparable outcome equation.
- Phase 1/Phase 2 tests.

Phase 4 estimators use the covariance-weighted GMM objective

```text
n * g_hat(alpha)' Sigma_hat(alpha)^(-1) g_hat(alpha)
```

with a small ridge added to the estimated moment covariance for numerical
stability. The older unweighted quadratic score remains only as a prototype
helper.

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
