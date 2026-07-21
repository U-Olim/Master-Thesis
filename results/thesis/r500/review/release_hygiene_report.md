# Phase 5 release-hygiene report

## Outcome

Final classification: **ready_with_owner_decisions**. Scientific and technical
integrity pass. The intended tag `thesis-r500-v1.0` is prepared but not created.

## Cross-platform protection

The minimal `.gitattributes` policy normalizes ordinary source and documentation
to LF, disables text conversion for the four immutable raw artifacts, and marks
PDF/PNG outputs binary. An isolated Git index configured with
`core.autocrlf=true` and native EOL preserved all four SHA-256 hashes. Manual
byte restoration was not required.

## Integrity

- Raw artifact contracts: pass
- Phase 1: pass
- Phase 2: pass
- Phase 3: pass (35 files; 190 consistency checks)
- Phase 4: pass (12 files; readiness unchanged)

## Repository hygiene and documents

The Phase 4-confirmed cache list is the only authorized cleanup scope; cleanup
is deliberately separate from this validator. `Experiments.qmd` and
`Experiments.pdf` are retained with warnings as design/calibration provenance,
not authoritative Phase 3 evidence. The PDF predates the QMD and equivalence
was not claimed.

## Owner decisions

No license or citation metadata was invented. Both remain owner decisions before
a public archival release. Their absence does not block private thesis
submission. Owner approval of the experimental-document disposition also
remains recommended.
