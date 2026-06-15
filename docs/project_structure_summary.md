# Project Architecture Summary

This repository uses a `src/` package layout. The main package is `ivqr_sim`.

- `config.py`: fixed thesis design constants.
- `dgp.py`: data-generating processes for DGP1, DGP2, and DGP3.
- `true_effects.py`: true structural quantile treatment effects.
- `moments.py`: IVQR and DML-IVQR moment functions.
- `inference.py`: confidence-region and inference routines.
- `metrics.py`: Monte Carlo performance metrics.
- `estimators/`: Full-control IVQR, Post-selection IVQR, and DML-IVQR.
- `simulation/`: design identifiers, runner orchestration, and aggregation.
- `reporting/`: final thesis tables and figures.
- `tests/`: verification tests for the package architecture and later logic.

Old prototype code is archived in `archive_old/`.
