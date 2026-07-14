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

`pixi run test` runs the fast suite and deselects tests marked `slow`.
`pixi run test_slow` runs only tests marked `slow`.

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

## Simulation Commands

### Development commands

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

`pixi run fast` and `pixi run full` retain the generic project defaults. In
particular, their DML defaults are not the tuned settings used for the final
thesis results.

Generic dry run:

```powershell
pixi run python scenarios/run_simulation.py --mode fast --dry-run
```

### Canonical final thesis production commands

The following tasks explicitly encode the final R=500 specification and write
to the canonical raw-result paths consumed by the analysis layer:

```powershell
pixi run final_oracle
pixi run final_post_selection
pixi run final_dml
```

These are expensive production simulations. They use full mode, 500
replications, the 21-point alpha grid on `[-1, 3]`, critical-value multiplier
`1.0`, and base seed `12345`. The post-selection task additionally uses Lasso
multiplier `1.8`; the DML task uses three folds, quantile penalty `0.07`, solver
`highs-ipm`, and ridge alpha `1.0`.

Inspect all three resolved configurations without running simulations:

```powershell
pixi run final_dry_run
```

The canonical tasks express complete replication indices 0--499. Equivalent
jobs may be split with `--rep-start` and `--rep-end`, using distinct block
outputs and manifests before assembling the canonical files. The repository
contains historical block-command examples, but its final provenance manifest
does not establish whether the canonical R=500 files were executed as single
runs or in blocks.

### Historical calibration and robustness commands

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

## Release Packaging

A clean repository archive should exclude local environments, caches, and test
artifacts: `.pixi/`, `__pycache__/`, `.pytest_tmp/`, `.pytest_cache/`, and
`.ruff_cache/`. These paths are already protected by `.gitignore`; packaging a
release does not require deleting the working environment.

## Final Results Analysis

Simulation and final analysis are separate workflows. After the completed R=500
files are present under `results/raw`, generate the thesis outputs with:

```powershell
pixi run results
```

This command loads and validates the oracle, post-selection, and DML result
files, then writes the final tables to `results/tables/` and figures to
`results/figures/`. It does not run simulations or alter the raw files.

These three CSVs are the canonical final R=500 inputs. The tracked
`results/raw/manifest.json` records their row and column counts, exact byte
sizes and SHA-256 hashes, and LF-normalized SHA-256 hashes. Exact hashes verify
byte-for-byte equality; canonical hashes allow equivalent LF and CRLF CSV
representations while substantive content changes still fail verification.
Final-run provenance stores the shared grid, seed, and critical-value settings
once under `common`, with estimator-specific tuning and canonical Pixi task
names under `estimators`. The post-selection Lasso multiplier is cross-checked
against its raw result column. The DML fold, penalty, solver, and ridge settings
are documented provenance because those values are not columns in the final DML
raw schema and therefore cannot be verified directly from that CSV.
`pixi run results` verifies the manifest schema, file identities, dimensions,
row totals, and hashes before analysis. Run `pixi run raw_manifest` only when
the canonical raw files intentionally change.

The overview tables summarize broad performance patterns. The canonical
`performance_by_design_cell.csv` and `performance_by_design_cell.tex` tables
preserve results by DGP, sample size, dimensionality, instrument strength,
quantile, and estimator.

## Final Raw Result Schema

All three completed R=500 result files contain:

- design identifiers: `dgp`, `n`, `p`, `pi`, `tau`, `rep`, and `seed`;
- the estimator label `estimator`;
- the point estimate `alpha_hat` and true parameter `alpha_true`;
- confidence-region hull bounds `cr_lower` and `cr_upper`, together with the
  reported accepted-set length `cr_length`;
- the coverage indicator `covered` and convergence indicator `converged`.

Grid-inverted confidence regions may be disconnected. Consequently,
`cr_length` may be smaller than `cr_upper - cr_lower`, and coverage cannot
always be inferred from hull inclusion alone. The post-selection file also
contains `n_selected_controls` and `selection_lasso_multiplier`.

Aggregate metrics such as bias, MAE, RMSE, and coverage probability are
computed by the final analysis workflow rather than stored as raw columns.

## Notes

The DML estimator is a DML-style residualized IVQR implementation. It uses
cross-fitting and residualized nuisance components, but it should not be read as
an exact density-weighted Chen-Huang-Tien DML-IVQR implementation.
