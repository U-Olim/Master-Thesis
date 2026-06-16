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

Current status: Phase 5A pilot runner implemented.

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

After editable installation, basic checks are:

```bash
pip install -e .
python -c "import ivqr_sim"
pytest -v
python scripts/01_smoke_test.py
python scripts/02_pilot_simulation.py --mode quick
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
