# Phase 2 deep scientific diagnostics of historical R=500 results

## 1. Scope and restrictions

Observed fact: this analysis reads only the three immutable historical artifacts. No simulation, source-data rewrite, estimator change, or inference change is part of Phase 2. Phase 1 definitions and outputs remain intact.

## 2. Source artifacts and hashes

oracle: `44ff5b1030e1a80386e62b840be9f842c2229f43e630000cc9e52567ac84be36` (22,624,466 bytes); post_selection: `7c9b914a8d5cf7cc7c3b9bda3458641de4f62bdfcf1644793871555275bfedd8` (30,818,201 bytes); dml: `27d3c514e7a608083647c9485e1ba4d73267cdf80753e3bae8c22a4ee4c5f803` (9,160,449 bytes); manifest: `bb9c0dfddc36e1d44be4749f7d1093313ce12040eab4754d5fa41ad42960b4c5` (7,340 bytes).

## 3. Warning analysis

Observed fact: no textual warning reasons are stored. The taxonomy is multi-label: affected-row counts and summed warning-event counts are both reported, and a row may belong to several categories.

oracle iteration-warning prevalence 0.7001 (50,409 rows; 93,657 events), affected valid coverage 0.9417 versus 0.9432 without, affected RMSE 0.8433 versus 0.8178, and affected mean CR length 2.4257 versus 2.3631; post_selection iteration-warning prevalence 0.8986 (64,697 rows; 237,048 events), affected valid coverage 0.9229 versus 0.9401 without, affected RMSE 0.8644 versus 0.8120, and affected mean CR length 2.3877 versus 2.3250. Statistical association is not a causal warning effect.

## 4. Empty and unresolved confidence regions

Observed fact: 99 exceptional rows were inspected: dml=43, oracle=13, post_selection=43. Oracle/Post-selection empty sets have valid empty components and no reversed bounds; unresolved rows remain separate from noncoverage; DML missing geometry is classified as legacy missingness, not numerical non-resolution.

Largest scenario-level exception cells: post_selection/dgp2/n=500/p=500/pi=0.1/tau=0.75: 5 empty_cr; post_selection/dgp1/n=500/p=200/pi=0.1/tau=0.25: 4 empty_cr; post_selection/dgp2/n=500/p=200/pi=0.1/tau=0.75: 4 empty_cr.

## 5. Paired estimator comparisons

All comparisons use identical design-replication keys and seeds. Differences are estimator A minus estimator B. Row-level paired uncertainty is reported overall; scenario files separately aggregate the 500 paired replications per design cell.

oracle minus dml: coverage -0.0068 (95% CI [-0.0089, -0.0046], m=71,944); oracle minus post_selection: coverage +0.0177 (95% CI [0.0158, 0.0196], m=71,952); post_selection minus dml: coverage -0.0243 (95% CI [-0.0267, -0.0220], m=71,915).

oracle minus dml: mean CR-length difference -0.5522; oracle minus post_selection: mean CR-length difference +0.0261; post_selection minus dml: mean CR-length difference -0.5777.

## 6. Coverage-uncertainty robustness

Both bounded Wald and Wilson 95% intervals use the actual conditional denominator. Wilson inclusion of 0.95 is reported scenario by scenario and in formal overall classification; no excluded row is restored to an expected denominator.

## 7. Formal classification rules

The non-compensatory rules are recorded in `classification_rules.json`; no weighted composite score is used. Results: dml: **acceptable with caveats** (diagnostic confidence limited; reasons ["failed: relative RMSE <= 1.15","failed: relative CR length <= 1.20","failed: rich diagnostic schema"]); oracle: **acceptable with caveats** (diagnostic confidence high; reasons ["failed: Wilson interval includes 0.95"]); post_selection: **concerning** (diagnostic confidence high; reasons ["overall coverage gap -0.0253 <= -0.02"]).

## 8. Estimator-specific findings

Oracle remains the infeasible precision benchmark but its overall Wilson interval does not include 0.95. Post-selection has the largest negative overall coverage gap. DML is closest to nominal coverage but has limited diagnostic confidence because its historical schema lacks warning, component, and resolution fields.

## 9. Cross-estimator conclusions

Statistical evidence: paired comparisons preserve the Phase 1 trade-off. DML covers more often than Oracle/Post-selection while using longer regions; this is an empirical coverage-length trade-off, not proof that interval length causes coverage. Oracle and Post-selection have closer error and length performance, but Post-selection covers less.

## 10. Limitations

Warning messages and failure reasons are absent; DML diagnostic metadata is absent; grid endpoints are not stored row-wise; and paired row-level intervals do not model cross-scenario heterogeneity. Associations cannot establish causal mechanisms.

## 11. Thesis implications

The two worst Post-selection scenarios are dgp2/n=500/p=500/pi=1/tau=0.75 (coverage 0.8380); dgp3/n=500/p=500/pi=1/tau=0.75 (coverage 0.8380). Stored evidence identifies concentration at the upper quantile and high dimension, with scenario warning rates and mean selected-control counts of 0.9680/26.5920; 0.8780/15.3680, respectively. One warning rate is above and one below the overall Post-selection warning prevalence, while selected-control complexity also differs. The stored fields therefore do not identify why the misses occurred. They must be presented as finite-sample weaknesses, not assigned a causal selection or warning explanation.

## 12. Exact reproduction command

```powershell
pixi run audit_r500_phase2
```
