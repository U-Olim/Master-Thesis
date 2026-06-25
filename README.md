# High-Dimensional IVQR under Weak Instruments

This repository contains the code for a master thesis Monte Carlo study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The project studies finite-sample performance of instrumental variable quantile regression estimators in high-dimensional designs with weak instruments.

The main simulation compares:

- **Oracle IVQR**: infeasible benchmark using the true active control set.
- **Post-selection IVQR**: feasible sparse-control benchmark using Lasso-based control selection.
- **DML-style IVQR**: residualized, cross-fitted IVQR estimator for high-dimensional controls.

A separate **full-control IVQR** benchmark is kept outside the main simulation because it uses all controls directly and becomes slow or unstable in high-dimensional settings.

---

## Repository structure

```text
src/
  config.py                 Compatibility import surface for project constants.
  dgp/                      Data-generating processes and true parameters.
  estimators/               Oracle, post-selection, DML-style, and full-control IVQR estimators.
  inference/                Alpha grids, moments, metrics, and confidence-region utilities.
  reporting/                Summary aggregation, tables, and figures.
  simulation/               Simulation configuration, runner, batching, chunking, validation.
  utils/                    Shared validation helpers.

scenarios/
  main_simulation.py        Main fast/full simulation runner.
  full_control_ivqr.py      Separate full-control IVQR benchmark runner.

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

---

## Environment

Pixi is the official environment manager for this project.

Install Pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then clone and install the project.

Check imports:

```bash
pixi run import_check
```

Run the normal non-slow test suite:

```bash
pixi run test_project
```

Run slow smoke/integration tests separately:

```bash
pixi run test_slow
```

---

## Simulation design

The main simulation varies sample size, control dimension, instrument strength, quantile index, and DGP.

```text
DGPs:                 dgp1, dgp2, dgp3
Sample sizes:         n = 500, 1000
Control dimensions:   p = 200, 500
Instrument strengths: pi = 1.0, 0.5, 0.25, 0.10
Quantiles:            tau = 0.25, 0.50, 0.75
Replications:         R = 10 in fast mode; R = 500 in full mode
Alpha grid:           9 points on [-1, 3]
DML folds:            K = 3
Estimators:           oracle, post_selection, dml
```

The 9-point alpha grid is a computational design choice. It keeps the IVQR grid-search Monte Carlo feasible while covering all true treatment effects in the implemented DGPs.

---

## Main workflow

Fast diagnostic run:

```bash
pixi run fast_mode
```

Main thesis simulation:

```bash
pixi run full_mode
```

Separate full-control benchmark:

```bash
pixi run full_control
```
