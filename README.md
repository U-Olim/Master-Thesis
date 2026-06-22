# High-Dimensional IVQR under Weak Instruments

This repository contains the code for a master thesis Monte Carlo study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The main thesis simulation compares:

- Oracle IVQR
- Post-selection IVQR
- DML-IVQR

Full-control IVQR is kept separate as an appendix-style benchmark.

## Environment

Pixi is the only official environment manager for this project.

Prerequisite:

- Install Pixi: <https://pixi.sh>

Setup:

```bash
git clone <YOUR_REPO_URL>
cd Master-Thesis
pixi install
```

Check the project:

```bash
pixi run test_project
```

## Main Workflow

Fast diagnostic run:

```bash
pixi run fast_mode
```

Main thesis simulation:

```bash
pixi run full_mode
```

Separate Full-Control IVQR benchmark:

```bash
pixi run full_control
```

Results, summaries, tables, and figures are generated automatically after each
normal run. No separate table or figure command is required.

## Outputs

Main simulation writes:

```text
results/raw/main_simulation_results.csv
results/summary/main_simulation_summary.csv
results/tables/main/
results/figures/main/
```

Full-control benchmark writes:

```text
results/raw/full_control_ivqr_results.csv
results/summary/full_control_ivqr_summary.csv
results/tables/full_control/
results/figures/full_control/
```

Generated tables include `comparison_table.csv`, `diagnostic_table.csv`,
`bias_wide.csv`, `rmse_wide.csv`, `coverage_wide.csv`, `cr_length_wide.csv`,
and `failure_rate_wide.csv`.

Generated figures include `fig_bias.png`, `fig_rmse.png`,
`fig_coverage.png`, `fig_cr_length.png`, and `fig_failure_rate.png`.

## Simulation Design

Main simulation:

- DGPs: `dgp1`, `dgp2`, `dgp3`
- Sample sizes: `n = 500, 1000`
- Control dimensions: `p = 200, 500`
- Instrument strengths: `pi = 1.0, 0.5, 0.25, 0.10`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 10` in fast mode, `R = 500` in full mode
- Main estimators: `oracle`, `post_selection`, `dml`

Separate full-control benchmark:

- Estimator: `full_control_ivqr`
- DGPs: `dgp1`
- Sample sizes: `n = 500, 1000`
- Control dimensions: `p = 20, 50, 100`
- Instrument strengths: `pi = 1.0`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 100`

## Useful Pixi Tasks

```bash
pixi run test_project
pixi run fast_mode
pixi run full_mode
pixi run full_control
pixi run clean_results
```

`test_project` runs the test suite and may take time. The simulation tasks are
computationally heavier, especially `full_mode` and `full_control`.

## Optional Reporting Utility

The normal workflow does not require a separate reporting step. The simulation
scripts aggregate results and write tables/figures automatically.

`scripts/03_make_tables.py` is kept only as an optional utility for regenerating
tables from existing raw or summary results.

## Notes

Full-control IVQR is intentionally excluded from the main simulation runner. It
is a separate naive benchmark and can be slow or unstable as the number of
controls grows.

Generated outputs are not committed. The `results/` directory is ignored and is
created only when simulations are run.
