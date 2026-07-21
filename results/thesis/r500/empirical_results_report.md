# Technical empirical-results report

## 1. Simulation design

The validated design contains 144 cells, 500 replications per cell, 72,000 rows per estimator, and 216,000 estimator rows overall. <!-- findings:simulation_design -->

## 2. Validation basis

Phase 1 supplies structural provenance and Phase 2 supplies scientific diagnostics; this package introduces no new inferential procedure. <!-- findings:simulation_design -->

## 3. Overall estimator performance

Oracle IVQR coverage 0.9421, RMSE 0.8357, mean CR length 2.4069; Post-selection IVQR coverage 0.9247, RMSE 0.8592, mean CR length 2.3813; DML-style IVQR coverage 0.9490, RMSE 0.9870, mean CR length 2.9591. <!-- findings:overall_oracle,overall_post_selection,overall_dml -->

## 4. Coverage

DML-style IVQR was closest to nominal coverage; Oracle and Post-selection coverage were lower. <!-- findings:overall_oracle,overall_post_selection,overall_dml -->

## 5. Bias and RMSE

Bias was -0.0067 for Oracle, -0.0566 for Post-selection, and 0.0093 for DML; corresponding RMSE values were 0.8357, 0.8592, and 0.9870. <!-- findings:bias_oracle,bias_post_selection,bias_dml,overall_oracle,overall_post_selection,overall_dml -->

## 6. Confidence-region length

Mean CR length was 2.4069 for Oracle, 2.3813 for Post-selection, and 2.9591 for DML. <!-- findings:overall_oracle,overall_post_selection,overall_dml -->

## 7. Paired estimator comparisons

Oracle IVQR had a paired coverage difference of +0.0177 relative to Post-selection IVQR. <!-- findings:paired_oracle_vs_post_selection -->

Oracle IVQR had a paired coverage difference of -0.0068 relative to DML-style IVQR. <!-- findings:paired_oracle_vs_dml -->

Post-selection IVQR had a paired coverage difference of -0.0243 relative to DML-style IVQR. <!-- findings:paired_post_selection_vs_dml -->

## 8. Quantile patterns

Oracle IVQR's lowest aggregated coverage by tau occurred at 0.75. <!-- findings:tau_pattern_oracle -->

Post-selection IVQR's lowest aggregated coverage by tau occurred at 0.75. <!-- findings:tau_pattern_post_selection -->

DML-style IVQR's lowest aggregated coverage by tau occurred at 0.5. <!-- findings:tau_pattern_dml -->

## 9. Instrument-strength patterns

Oracle IVQR's lowest aggregated coverage by pi occurred at 1.0. <!-- findings:pi_pattern_oracle -->

Post-selection IVQR's lowest aggregated coverage by pi occurred at 1.0. <!-- findings:pi_pattern_post_selection -->

DML-style IVQR's lowest aggregated coverage by pi occurred at 1.0. <!-- findings:pi_pattern_dml -->

## 10. Weak scenarios

The five weakest coverage scenarios per estimator are selected mechanically from the Phase 2 ranking and reported with their validated uncertainty intervals. <!-- findings:weak_oracle_1,weak_oracle_2,weak_oracle_3,weak_oracle_4,weak_oracle_5,weak_post_selection_1,weak_post_selection_2,weak_post_selection_3,weak_post_selection_4,weak_post_selection_5,weak_dml_1,weak_dml_2,weak_dml_3,weak_dml_4,weak_dml_5 -->

## 11. Warning diagnostics

Warnings were associated with the reported outcomes for Oracle IVQR. <!-- findings:diagnostic_oracle -->

Warnings were associated with the reported outcomes for Post-selection IVQR. <!-- findings:diagnostic_post_selection -->

## 12. Empty and unresolved regions

Validated exception counts were 13 for Oracle, 43 for Post-selection, and 43 DML legacy missing-geometry rows. <!-- findings:diagnostic_oracle,diagnostic_post_selection,diagnostic_dml -->

## 13. Estimator classifications

Oracle IVQR: acceptable with caveats; Post-selection IVQR: concerning; DML-style IVQR: acceptable with caveats. <!-- findings:overall_oracle,overall_post_selection,overall_dml -->

## 14. Limitations

Historical Post-selection results use multiplier 1.0; no conclusion about 1.8 follows. <!-- findings:historical_multiplier -->

The historical DML schema has 15 columns; unavailable warning, unresolved, and rich geometry diagnostics are not zeros. <!-- findings:diagnostic_dml -->

## 15. Defensible conclusions

DML's higher coverage coincided with longer regions and higher RMSE; Oracle offered the lowest RMSE; Post-selection displayed the largest undercoverage. Coverage, error, and interval length must be interpreted jointly. <!-- findings:overall_oracle,overall_post_selection,overall_dml -->
