# High-Dimensional IVQR Monte Carlo Study

This repository contains thesis code for Monte Carlo simulations of instrumental
variable quantile regression under high-dimensional controls and weak
instruments.

The project compares three estimators:

- Oracle IVQR
- Post-selection IVQR
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
  analysis/     Final-result validation, tables, and figures.
  dgp/          Simulation designs and data-generating processes.
  estimators/   Oracle, post-selection, and DML estimators.
  ivqr/         Alpha grids, CH inverse-IVQR, moments, and confidence regions.
  simulation/   Simulation config, runner, and result-row construction.
  utils/        Validation and timing helpers.

documents/
  Experiments.qmd  Experiment-design and simulation-run documentation.
  Experiments.pdf  Rendered experiment-design document.

scripts/
  make_results.py

tests/          Unit and integration tests for simulation and analysis code.
```

## Environment

Pixi is the only project manager used by this repository.

```powershell
pixi run fast
pixi run full
pixi run import_check
pixi run results
pixi run test
pixi run test_slow
```

`test_slow` may deselect all tests when no tests are marked `slow`; this is
expected until long-running production checks are added.

## Simulation Modes

- `fast`: `R = 10`
- `full`: `R = 500`

Default estimators are `oracle post_selection dml`.

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
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

Full default with explicit output:

```powershell
pixi run python scenarios/run_simulation.py --mode full --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/full_results.csv --manifest results/raw/full_manifest.json
```

Full R500 runs can be split into replication blocks with `--rep-start` and
`--rep-end`. `--reps` remains the total planned replication count, while the
block arguments select the global replication indices for the current process.
Each block should use a separate output and manifest file.

Oracle R500 block 0-99:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle --reps 500 --rep-start 0 --rep-end 99 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/oracle_R500_grid41_block000_099.csv --manifest results/raw/oracle_R500_grid41_block000_099_manifest.json
```

Post-selection R500 block 0-99:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --reps 500 --rep-start 0 --rep-end 99 --selection-lasso-multiplier 1.8 --alpha-min -2 --alpha-max 4 --alpha-grid-size 41 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/post_selection_R500_lasso180_grid41_block000_099.csv --manifest results/raw/post_selection_R500_lasso180_grid41_block000_099_manifest.json
```

Fast DML only:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --n-jobs 4 --batch-size 10 --alpha-grid-size 21 --output results/raw/fast_dml.csv --manifest results/raw/fast_dml_manifest.json
```

DML chosen experimental setting:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators dml --reps 5 --dml-k-folds 3 --dml-quantile-penalty 0.07 --dml-quantile-solver highs-ipm --alpha-min -1 --alpha-max 3 --alpha-grid-size 21 --base-seed 12345 --n-jobs 8 --batch-size 10 --output results/raw/dml_R5_penalty007_solver_highsipm.csv --manifest results/raw/dml_R5_penalty007_solver_highsipm_manifest.json
```

For parallel DML batches, cap numerical-library threads to avoid process/thread
oversubscription:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Baseline post-selection:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --base-seed 12345 --output results/raw/fast_post_selection.csv --manifest results/raw/fast_post_selection_manifest.json
```

Dry run:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --dry-run
```

## Post-selection Lasso Multiplier

`--selection-lasso-multiplier` controls only the Lasso control-selection step
inside post-selection IVQR. The default is `1.0`, which preserves baseline
behavior. It is separate from `--critical-value-multiplier`, which affects
confidence-region inversion.

Values above `1.0` increase the Lasso penalty selected by cross-validation and
therefore tend to select fewer controls. This can reduce over-selection, but
large values can under-select controls and introduce omitted-control bias.

Post-selection with a 1.2 multiplier:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --estimators post_selection --selection-lasso-multiplier 1.2 --base-seed 12345 --output results/raw/fast_post_selection_lasso120.csv --manifest results/raw/fast_post_selection_lasso120_manifest.json
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
pixi run python scenarios/run_simulation.py --mode fast --estimators oracle post_selection dml --base-seed 12345 --output results/raw/fast_all.csv --manifest results/raw/fast_all_manifest.json
```

## Output Folders

- `results/raw`: raw estimator-level CSV files and run manifests
- `results/tables`: final thesis tables in CSV and LaTeX formats
- `results/figures`: final thesis figures in PDF and PNG formats

Simulation outputs are ignored by git except for `.gitkeep` placeholders.

## Final Results Analysis

Simulation and final analysis are separate workflows. After the completed R=500
files are present under `results/raw`, generate the thesis outputs with:

```powershell
pixi run results
```

This command loads and validates the oracle, post-selection, and DML result
files, then writes the final tables to `results/tables/` and figures to
`results/figures/`. It does not run simulations or alter the raw files.

The overview tables summarize broad performance patterns. The canonical
`performance_by_design_cell.csv` and `performance_by_design_cell.tex` tables
preserve results by DGP, sample size, dimensionality, instrument strength,
quantile, and estimator.

## Key Diagnostics

Raw results include diagnostics for:

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
