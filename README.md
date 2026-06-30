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
pixi run fast
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
pixi run fast
pixi run full
pixi run import_check
pixi run test
pixi run test_slow
```

## Simulation Modes

- `fast`: `R = 10`
- `full`: `R = 500`

Default estimators are `oracle post_selection dml`. `full_control` is available
but must be requested explicitly because it can be slow when `p` is large.

## Design Grid

| DGP | Role | Active controls |
|---|---|---:|
| `dgp1` | Baseline sparse Gaussian design | 5 |
| `dgp2` | Denser sparse selection-stress design | 10 |
| `dgp3` | Heavy-tail sparse robustness design | 5 |

The reduced sparsity keeps the baseline design cleaner while high dimensionality
is maintained through `p = 200` and `p = 500`. Weak-instrument difficulty is
still controlled by `pi`.

## Main Commands

Fast default:

```powershell
pixi run fast
```

Full default:

```powershell
pixi run full
```

Fast all estimators:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection full_control dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

Full default with explicit output:

```powershell
pixi run python scenarios/run_simulation.py --mode full --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/full_results.csv --manifest results/raw/full_manifest.json
```

Fast DML only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

Dry run:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --dry-run
```

## Reproducibility and Separate Estimator Runs

The project uses a fixed default base seed, `12345`. Each design cell has a
deterministic seed derived from `base_seed`, `dgp`, `n`, `p`, `pi`, `tau`, and
`rep`.

The design seed does not depend on estimator name, estimator order, number of
workers, or batch size. Therefore, estimators can be run separately on different
PCs and later merged, as long as they use the same mode, design settings, and
base seed. These runs generate identical data for matching design cells.

Oracle only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle --base-seed 12345 --output results/raw/fast_oracle.csv --manifest results/raw/fast_oracle_manifest.json
```

DML only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --base-seed 12345 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

All estimators:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection full_control dml --base-seed 12345 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
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
