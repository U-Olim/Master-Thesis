# Joint scientific audit of immutable R=500 results

## 1. Executive conclusion

All three files satisfy the recorded structural and cross-estimator alignment contracts. They are suitable for thesis analysis subject to the estimator-specific coverage and finite-sample caveats below. Oracle is an infeasible benchmark; Post-selection reflects estimated-support costs; DML uses a different inferential construction and has fewer historical diagnostics.

## 2. Artifact integrity

Manifest checksum, byte-size, ordered-column, and dimension validation passed. oracle: SHA-256 `44ff5b1030e1a80386e62b840be9f842c2229f43e630000cc9e52567ac84be36`, 22,624,466 bytes, 72,000 rows, 43 columns, modified 2026-07-21T08:44:27.612480+00:00; post_selection: SHA-256 `7c9b914a8d5cf7cc7c3b9bda3458641de4f62bdfcf1644793871555275bfedd8`, 30,818,201 bytes, 72,000 rows, 52 columns, modified 2026-07-21T09:09:28.858644+00:00; dml: SHA-256 `27d3c514e7a608083647c9485e1ba4d73267cdf80753e3bae8c22a4ee4c5f803`, 9,160,449 bytes, 72,000 rows, 15 columns, modified 2026-07-13T14:10:13.506604+00:00; manifest: SHA-256 `bb9c0dfddc36e1d44be4749f7d1093313ce12040eab4754d5fa41ad42960b4c5`, 7,340 bytes, non-tabular JSON, modified 2026-07-21T11:36:30.983154+00:00.

## 3. Design completeness

Each estimator contains 72,000 rows, 144 design cells, 500 replications numbered 0-499, and 144 rows per replication. Natural keys, seeds, and `alpha_true` agree exactly across estimators. Oracle is in natural-key order; historical Post-selection and DML file order is deterministic but not lexicographic natural-key order. Audit products are stably sorted; raw files were not reordered.

## 4. Metric definitions

Bias and RMSE use finite point-estimation errors. Confidence-region length uses finite historical `cr_length` values. Unsupported diagnostics remain missing, not zero.

## 5. Coverage denominator policy

Unconditional coverage uses all nonmissing Boolean `covered` observations. Conditional coverage additionally requires convergence, finite CR geometry, and, for Oracle/Post-selection, historical numerical resolution. DML has no resolution field, so its denominator requires convergence and finite legacy CR fields. Unresolved observations are reported separately and never coded as noncoverage.

Nominal coverage is 95%, confirmed by production estimator `confidence_level` defaults. Scenario Monte Carlo standard errors use sqrt(p(1-p)/m), with bounded Wald 95% intervals and the actual denominator. A scenario is within Monte Carlo uncertainty when its interval contains 95%; otherwise a negative gap of at least 10 percentage points is severe, a smaller negative gap is moderate, and a positive statistically distinguishable gap is overcoverage.

## 6. Oracle results

Classification: **strong**. Conditional coverage is 0.9421 (m=71,987); unconditional coverage is 0.9419 (m=72,000); bias is -0.0067; RMSE is 0.8357; mean CR length is 2.4069; convergence is 1.0000. The conditional rule excludes 13 rows. There are 0 severe-undercoverage scenarios under the disclosed rule.
Full-grid frequency is 0.1937, numerical resolution is 1.0000, unresolved rows are 0, and warning frequency is 0.7001. True-support knowledge does not remove weak-instrument finite-sample uncertainty.

## 7. Post-selection results

Classification: **acceptable with caveats**. Conditional coverage is 0.9247 (m=71,957); unconditional coverage is 0.9241 (m=72,000); bias is -0.0566; RMSE is 0.8592; mean CR length is 2.3813; convergence is 1.0000. The conditional rule excludes 43 rows. There are 2 severe-undercoverage scenarios under the disclosed rule.
Mean selected controls are 18.9380 (median 17.0000; range 2-108). Unresolved rows are 2, full-grid frequency is 0.1822, and warning frequency is 0.8986. The multiplier record is `1:72000`: these historical results use 1.0, not the future-run value 1.8; no counterfactual claim is made.

## 8. DML results

Classification: **strong**. Conditional coverage is 0.9490 (m=71,957); unconditional coverage is 0.9484 (m=72,000); bias is 0.0093; RMSE is 0.9870; mean CR length is 2.9591; convergence is 1.0000. The conditional rule excludes 43 rows. There are 0 severe-undercoverage scenarios under the disclosed rule.
The 15-column DML artifact contains no CR components, resolution status, block count, warning, or nuisance/cross-fitting diagnostics. Those properties cannot be audited or reported as zero. Its 43 finite-CR exclusions are not called unresolved because no resolution field exists.

## 9. Cross-estimator comparison

Pairwise scenario differences are in `comparison_checks.csv`. Coverage, RMSE, and interval length must be interpreted jointly: lower RMSE does not establish superior inference, and wider intervals may purchase coverage. Oracle is a benchmark, not a feasible competitor; Post-selection differences reflect estimated-support costs; DML diagnostic differences are not directly comparable where fields are absent.

## 10. Weak-instrument patterns

dml absolute_bias: 80.6% follow the stated direction; dml conditional_coverage: 43.5% follow the stated direction; dml mean_cr_length: 100.0% follow the stated direction; dml nonconvergence_rate: 100.0% follow the stated direction; dml rmse: 100.0% follow the stated direction; oracle absolute_bias: 77.8% follow the stated direction; oracle conditional_coverage: 36.1% follow the stated direction; oracle full_grid_rate: 100.0% follow the stated direction; oracle mean_cr_length: 100.0% follow the stated direction; oracle nonconvergence_rate: 100.0% follow the stated direction; oracle rmse: 100.0% follow the stated direction; oracle unresolved_rate: 100.0% follow the stated direction; post_selection absolute_bias: 81.5% follow the stated direction; post_selection conditional_coverage: 38.9% follow the stated direction; post_selection full_grid_rate: 100.0% follow the stated direction; post_selection mean_cr_length: 100.0% follow the stated direction; post_selection nonconvergence_rate: 100.0% follow the stated direction; post_selection rmse: 100.0% follow the stated direction; post_selection unresolved_rate: 100.0% follow the stated direction.

## 11. Sample-size patterns

dml absolute_bias: 63.9% follow the stated direction; dml conditional_coverage: 58.3% follow the stated direction; dml mean_cr_length: 94.4% follow the stated direction; dml nonconvergence_rate: 100.0% follow the stated direction; dml rmse: 94.4% follow the stated direction; oracle absolute_bias: 69.4% follow the stated direction; oracle conditional_coverage: 52.8% follow the stated direction; oracle full_grid_rate: 100.0% follow the stated direction; oracle mean_cr_length: 97.2% follow the stated direction; oracle nonconvergence_rate: 100.0% follow the stated direction; oracle rmse: 97.2% follow the stated direction; oracle unresolved_rate: 100.0% follow the stated direction; post_selection absolute_bias: 83.3% follow the stated direction; post_selection conditional_coverage: 61.1% follow the stated direction; post_selection full_grid_rate: 100.0% follow the stated direction; post_selection mean_cr_length: 95.8% follow the stated direction; post_selection nonconvergence_rate: 100.0% follow the stated direction; post_selection rmse: 100.0% follow the stated direction; post_selection unresolved_rate: 97.2% follow the stated direction.

## 12. Dimension patterns

dml absolute_bias: 62.5% follow the stated direction; dml conditional_coverage: 50.0% follow the stated direction; dml mean_cr_length: 91.7% follow the stated direction; dml nonconvergence_rate: 100.0% follow the stated direction; dml rmse: 88.9% follow the stated direction; oracle absolute_bias: 47.2% follow the stated direction; oracle conditional_coverage: 55.6% follow the stated direction; oracle full_grid_rate: 69.4% follow the stated direction; oracle mean_cr_length: 45.8% follow the stated direction; oracle nonconvergence_rate: 100.0% follow the stated direction; oracle rmse: 41.7% follow the stated direction; oracle unresolved_rate: 100.0% follow the stated direction; post_selection absolute_bias: 70.8% follow the stated direction; post_selection conditional_coverage: 61.1% follow the stated direction; post_selection full_grid_rate: 72.2% follow the stated direction; post_selection mean_cr_length: 56.9% follow the stated direction; post_selection mean_selected_controls: 94.4% follow the stated direction; post_selection nonconvergence_rate: 100.0% follow the stated direction; post_selection rmse: 65.3% follow the stated direction; post_selection unresolved_rate: 98.6% follow the stated direction.

## 13. Quantile patterns

dml: lowest mean scenario coverage at tau=0.5 (0.9449); largest mean RMSE at tau=0.25 (0.9912); oracle: lowest mean scenario coverage at tau=0.75 (0.9336); largest mean RMSE at tau=0.75 (0.7711); post_selection: lowest mean scenario coverage at tau=0.75 (0.9050); largest mean RMSE at tau=0.75 (0.8217).

## 14. DGP patterns

dml: lowest mean scenario coverage at dgp=dgp1 (0.9480); largest mean RMSE at dgp=dgp2 (0.9770); oracle: lowest mean scenario coverage at dgp=dgp2 (0.9383); largest mean RMSE at dgp=dgp2 (0.7975); post_selection: lowest mean scenario coverage at dgp=dgp2 (0.9140); largest mean RMSE at dgp=dgp2 (0.8320). DGP1 is the five-control Gaussian baseline, DGP2 is the denser ten-control Gaussian selection-stress design, and DGP3 is the five-control heavy-tail robustness design. These are descriptive associations, not causal explanations.

## 15. Worst scenarios

`worst_scenarios.csv` contains 140 metric-labelled ranked rows: ten per estimator for coverage, RMSE, CR length, non-convergence, and resolution where available. Three worst coverage gaps per estimator: dml: dgp1/n=1000/p=500/pi=0.5/tau=0.75: gap -0.032 (m=500), dgp1/n=500/p=200/pi=1/tau=0.25: gap -0.024 (m=500), dgp1/n=500/p=200/pi=1/tau=0.5: gap -0.022 (m=500); oracle: dgp2/n=500/p=200/pi=0.25/tau=0.75: gap -0.040 (m=500), dgp3/n=500/p=500/pi=1/tau=0.75: gap -0.038 (m=500), dgp2/n=500/p=500/pi=0.25/tau=0.75: gap -0.036 (m=500); post_selection: dgp2/n=500/p=500/pi=1/tau=0.75: gap -0.112 (m=500), dgp3/n=500/p=500/pi=1/tau=0.75: gap -0.112 (m=500), dgp2/n=500/p=200/pi=1/tau=0.75: gap -0.092 (m=500).

## 16. Suspicious findings

dml/not_natural_key_sorted: 1; post_selection/not_natural_key_sorted: 1.

The row-order flags document historical file order, not corruption. Component-based coverage and geometry validation passed for Oracle and Post-selection. DML has no components, so false coverage inside its hull is not treated as inconsistent because the accepted set may be disconnected.

## 17. Thesis defensibility

The rule is: strong requires overall conditional coverage >=93%, convergence >=99%, and no more than 10% severe-undercoverage scenarios; acceptable with caveats requires coverage >=85% and convergence >=95%; problematic requires coverage >=70% and convergence >=80%; otherwise not defensible. Scenario weaknesses must still be disclosed. The classifications describe these artifacts, not universal method validity.

## 18. Required caveats

Oracle is infeasible; weak instruments can generate broad or full-grid regions; Post-selection uses multiplier 1.0 and estimated support; DML's historical schema prevents claims beyond its 15 fields; 500 replications leave measurable Monte Carlo uncertainty. Difficult-scenario weakness is not evidence of corruption or, by itself, implementation failure.

## 19. Limitations of historical schemas

This audit does not prove or disprove asymptotic validity or identify causal reasons for every pattern. Oracle's 43-column and DML's 15-column historical schemas differ from current future-output schemas and were validated as historical contracts rather than rewritten.

## 20. Recommendation for thesis tables and figures

Later thesis outputs should show conditional coverage with denominators and Monte Carlo uncertainty, RMSE and CR length together, weak-instrument panels, ranked caveat scenarios, and explicit schema/calibration notes. This audit does not create final publication tables or figures.
