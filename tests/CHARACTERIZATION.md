# Behavioral characterization coverage

The reference fixtures are pinned to `pre-refactor-r500`. Floating-point
comparisons use `rtol=0, atol=1e-12`; identifiers, statuses, schemas, supports,
and serialized structure use exact equality.

| Behavior | Existing coverage before this layer | Assessment | Added coverage |
|---|---|---|---|
| Seed determinism and global replication IDs | `test_runner.py` seed/grid tests | Partial: no exact reference seed | Exact reference seed and worker/block invariance |
| DGP reproducibility and support | `test_dgp.py` | Partial: no frozen numeric values | Compact deterministic DGP fixture |
| Estimator return contracts | `test_estimators.py` | Partial: type/status only | Frozen Oracle, Post-selection, and DML scientific outputs |
| Confidence-region geometry and tri-state coverage | `test_ivqr.py` | Adequate | Reused without duplication |
| Adaptive midpoint/refinement ordering and limits | `test_ivqr.py` | Adequate | Reused without duplication |
| Warning and hard-failure policies | `test_ivqr.py`, `test_estimators.py` | Adequate | Reused without duplication |
| Exact serializer schemas | estimator output tests | Adequate for synthetic rows | Added real internal-row preservation check |
| Serial/parallel equivalence | None | Missing | Tiny single-estimator comparisons for all three estimators |
| Block/uninterrupted equivalence | Synthetic merge tests only | Partial | Production-run Oracle blocks compared with uninterrupted output |
| Resume/uninterrupted equivalence | Manifest and append mechanics only | Partial | Scientific output comparison after resume |
| Runner/direct-estimator equivalence | Oracle support propagation only | Partial | All three estimators on the reference design |
| Removed full-mode behavior | Full mode previously shared one DGP draw across three estimators and emitted a 150-column union CSV | Replaced intentionally | Generic/full and internal multi-estimator execution are rejected before output; separate runners retain the same seed-to-DGP mapping |

The large historical R=500 CSVs are deliberately not fixtures for this layer.
