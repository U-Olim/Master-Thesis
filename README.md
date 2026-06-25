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
successful fast, full, or full-control run. No separate make-table or figure
script is required. Dry-runs print the planned settings without creating raw
results, manifests, tables, or figures.

## Outputs

Fast mode writes:

```text
results/raw/fast_mode_results.csv
results/summary/fast_mode_summary.csv
results/tables/fast/
results/figures/fast/
```

Full mode writes:

```text
results/raw/full_mode_results.csv
results/summary/full_mode_summary.csv
results/tables/full/
results/figures/full/
```

Full-control benchmark writes:

```text
results/raw/full_control_ivqr_results.csv
results/summary/full_control_ivqr_summary.csv
results/tables/full_control/
results/figures/full_control/
```

When `--resume` is used, new pending rows are appended to the existing raw CSV.
Use the same manifest path to guard against accidentally resuming with
incompatible settings.

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
- Alpha grid size: `9`
- Main estimators: `oracle`, `post_selection`, `dml`

Separate full-control benchmark:

- Estimator: `full_control_ivqr`
- DGPs: `dgp1`
- Sample sizes: `n = 500, 1000`
- Control dimensions: `p = 20, 50, 100`
- Instrument strengths: `pi = 1.0`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 500`
- Alpha grid size: `9`

## Useful Pixi Tasks

```bash
pixi run test_project
pixi run fast_mode
pixi run full_mode
pixi run full_control
pixi run import_check
pixi run test_slow
```

`test_project` runs the test suite and may take time. The simulation tasks are
computationally heavier, especially `full_mode` and `full_control`.

## Notes

Full-control IVQR is intentionally excluded from the main simulation runner. It
is a separate naive benchmark and can be slow or unstable as the number of
controls grows.

Generated outputs are not committed. The `results/` directory is ignored and is
created only when simulations are run.
