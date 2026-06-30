# High-Dimensional IVQR Monte Carlo Study

This repository contains thesis code for Monte Carlo simulations of instrumental
variable quantile regression under high-dimensional controls and weak
instruments.

The project compares four estimators:

- Oracle IVQR
- Post-selection IVQR
- Full-control IVQR
- DML-style residualized IVQR

The only simulation entry point is:

```powershell
pixi run python scenarios/run_simulation.py --mode fast
```

## Structure

```text
scenarios/
  run_simulation.py

src/
  dgp/          Simulation designs and data-generating processes.
  estimators/   Oracle, post-selection, full-control, and DML estimators.
  ivqr/         Alpha grids, CH inverse-IVQR, moments, and confidence regions.
  simulation/   Simulation config, runner, and result-row construction.
  reporting/    Summary aggregation, tables, and figures.
  utils/        Validation and timing helpers.

reports/
  monte_carlo_runs_summary.qmd

tests/
  test_dgp.py
  test_estimators.py
  test_ivqr.py
  test_runner.py
  test_reporting.py
```

## Environment

Pixi is the only project manager used by this repository.

```powershell
pixi run import_check
pixi run test
```

## Simulation Modes

- `fast`: `R = 10`
- `full`: `R = 500`

Default estimators are `oracle post_selection dml`. `full_control` is available
but must be requested explicitly because it can be slow when `p` is large.

## Main Commands

Fast default:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_results.csv --manifest results/raw/fast_manifest.json
```

Fast DML only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

Fast full-control only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators full_control --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_full_control.csv --manifest results/raw/fast_full_control_manifest.json
```

Fast all estimators:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection full_control dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

Full default:

```powershell
pixi run python scenarios/run_simulation.py --mode full --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/full_results.csv --manifest results/raw/full_manifest.json
```

Tiny smoke:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators full_control --reps 1 --dgps dgp1 --n-values 80 --p-values 5 --pi-values 1.0 --taus 0.5 --max-designs 1 --n-jobs 1 --alpha-grid-size 5 --no-reports --output results/raw/smoke_full_control.csv --manifest results/raw/smoke_full_control_manifest.json
```

Tiny all-estimator smoke:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection full_control dml --reps 1 --dgps dgp1 --n-values 80 --p-values 20 --pi-values 1.0 --taus 0.5 --max-designs 1 --n-jobs 1 --alpha-grid-size 5 --no-reports --output results/raw/smoke_all_estimators.csv --manifest results/raw/smoke_all_estimators_manifest.json
```

Dry run:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --dry-run
```

## Output Folders

- `results/raw`: raw estimator-level CSV files and run manifests
- `results/summary`: aggregated summary CSV files
- `results/tables`: thesis-ready CSV tables
- `results/figures`: generated diagnostic figures

Simulation outputs are ignored by git except for `.gitkeep` placeholders.

## Key Diagnostics

Raw results and summaries include diagnostics for:

- coverage
- confidence-region length
- boundary hits
- disconnected confidence regions
- failed-alpha rate
- RMSE
- MAE
- runtime

## Notes

The DML estimator is a DML-style residualized IVQR implementation. It uses
cross-fitting and residualized nuisance components, but it should not be read as
an exact density-weighted Chen-Huang-Tien DML-IVQR implementation.
