# Full-run IVQR analysis report

This report is generated deterministically from the existing simulation CSVs. No
simulation or estimator implementation is invoked.

## Reproduction

Run this exact one-line Windows PowerShell command from the repository root:

```powershell
pixi run python scripts/report_full_run.py
```

If an alternate Oracle file is supplied and its name contains parentheses, quote
it, for example: `--oracle "results\raw\oracle_ivqr(1).csv"`.

## Inputs and panel validation

- DML-IVQR: `results\raw\dml_ivqr.csv`
- Oracle IVQR: `results\raw\oracle_ivqr.csv`
- Post-selection IVQR: `results\raw\post_selection_ivqr.csv`

Each estimator has 72,000 rows across
144 design cells, with
500 replications per
cell. Design variables are `dgp`, `n`, `p`, `pi`, and `tau`; `rep` is the
replication identifier. The design-replication keys and seeds are identical
across estimators, and no duplicates were found.

Observed design values are: `dgp` = dgp1, dgp2, dgp3; `n` = 500, 1000; `p` = 200, 500; `pi` = 0.1, 0.25, 0.5, 1.0; `tau` = 0.25, 0.5, 0.75. The DML source has
15 columns, Oracle has
43, and
post-selection has
52.
`validation.json` records every original column and every column unavailable for
each estimator.

## Coverage and status rules

Primary empirical coverage uses only resolved replications. Explicit
`cr_is_numerically_resolved` values define the denominator for Oracle and
post-selection. `partially_unresolved` and `fully_unresolved` rows have missing
analysis coverage and are reported separately; their raw `covered` values are
never silently treated as success or failure.

`empty_valid` is an explicitly resolved empty set. It contributes zero CR length
and is uncovered because its validated component list contains no true parameter.
In contrast, a missing DML CR is **not** called empty: the legacy DML schema has no
components, CR status, or numerical-resolution flag. For DML only, a complete CR
triplet plus estimator convergence is the observable resolved proxy. Its
43 missing CR triplets
are classified `missing_status_unavailable`, excluded from primary coverage, and
reported as unresolved. DML empty- and disconnected-region rates remain `NA`.

Confidence-region length is total component length for detailed schemas; resolved
empty regions use length zero. Full-grid and boundary diagnostics use the
simulation grid [-1, 3], as configured for these full runs.
Unavailable DML warning, rank, refinement, and selected-control diagnostics remain
`NA` rather than being fabricated as zero.

## Overall results

Coverage quantities in this display are percentages; MC intervals use the normal
Monte Carlo approximation `coverage +/- 1.96 * MCSE`.

| estimator_label | replications | resolved_replications | empirical_coverage | coverage_mcse | coverage_mc95_lower | coverage_mc95_upper | bias | mae | rmse | mean_cr_length | unresolved_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DML-IVQR | 72000 | 71957 | 94.8983 | 0.0820254 | 94.7376 | 95.0591 | 0.0093 | 0.759565 | 0.986971 | 2.95914 | 0.0597222 |
| Oracle IVQR | 72000 | 72000 | 94.1944 | 0.0871502 | 94.0236 | 94.3653 | -0.00674167 | 0.60181 | 0.835711 | 2.4065 | 0 |
| Post-selection IVQR | 72000 | 71998 | 92.4151 | 0.0986704 | 92.2217 | 92.6085 | -0.0566444 | 0.623675 | 0.859228 | 2.37994 | 0.00277778 |

## Benchmark reconciliation

- DML-IVQR: primary 94.8983%; reference 94.8%; difference +0.098 percentage points.
- Oracle IVQR: primary 94.1944%; reference 94.2%; difference -0.006 percentage points.
- Post-selection IVQR: primary 92.4151%; reference 92.4%; difference +0.015 percentage points.

All three primary coverage values are within the predeclared 0.5-percentage-point
reconciliation tolerance. DML's unconditional raw mean is
94.8417%; the
small difference from primary coverage comes from the 43 status-unavailable rows.
Post-selection has two explicitly unresolved rows, one of which has raw
`covered=True`; it is correctly excluded from both sides of primary coverage.

## Generated files

- `combined_standardized_results.csv`
- `table_01_overall.csv`
- `table_02_by_quantile.csv`
- `table_03_by_strength.csv`
- `table_04_by_n_p.csv`
- `table_05_by_design_cell.csv`
- `table_06_worst_cells.csv`
- `table_07_diagnostics.csv`
- `validation.json`
- `analysis_report.md`

`combined_standardized_results.csv` retains all source columns and adds derived,
tri-state analysis fields. Summary-table rates are `NA` when an estimator's source
schema does not contain the needed diagnostic.

## Remaining data-quality limitations

- The legacy DML schema cannot distinguish an empty valid set from a numerical
  failure or another cause of missing CR geometry.
- DML has no component representation, so disconnected regions cannot be
  reconstructed and its observed hull is the only available CR geometry.
- DML has no iteration-warning, rank-failure, refinement-limit, or variable-
  selection diagnostics. These are preserved as unavailable.
- The normal 95% Monte Carlo interval is a simulation-uncertainty interval for
  empirical coverage, not a confidence interval for an individual estimate.
