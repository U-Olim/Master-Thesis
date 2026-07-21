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

## Output schema ownership

Internal simulation rows remain rich 150-column diagnostic objects. Final
Oracle, Post-selection, and DML CSVs are projected through separate serializers
onto the immutable schemas registered in `simulation.output_schemas`.
Historical thesis artifacts may retain older validated schemas and are handled
by the analysis compatibility layer. Any schema change therefore requires an
explicit compatibility path and exact regression coverage for column order,
retained values, nulls, statuses, and serialized confidence-region components.

## Configuration ownership

Immutable typed configurations separate shared execution, design, and alpha-grid
settings from estimator-owned inference settings. Oracle owns CH inference;
Post-selection owns CH inference plus its selection multiplier; DML owns only its
cross-fitting and nuisance settings. Dedicated parsers compose only the relevant
argument groups. The generic single-estimator parser remains a compatibility
layer and preserves its historical CLI, validation, manifest, and resume payload.

## Execution infrastructure ownership

`simulation.seeds` owns the immutable SHA-256 seed mapping, while
`simulation.designs` owns grid validation, enumeration order, replication blocks,
and design keys. `simulation.dispatch` owns estimator names and estimator-specific
argument dispatch. `simulation.execution` owns one-design execution, failure-row
conversion, worker arguments, serial/parallel execution, and deterministic
parallel sorting. `simulation.resume` owns completion filtering, and
`simulation.persistence` owns output projection, append validation, and CSV
writing. `simulation.runner` remains the stable compatibility facade and
re-exports established public names. CLI manifest validation and progress
reporting remain in `scenarios.run_simulation`; block merging remains separate.
