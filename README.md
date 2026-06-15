# High-Dimensional IVQR under Weak Instruments

This repository contains the code architecture for a master thesis simulation study:

**High-Dimensional Instrumental Variable Quantile Regression under Weak Instruments: A Monte Carlo Study**

The final project will compare three IVQR-type estimators:

- Full-control IVQR
- Post-selection IVQR
- DML-IVQR

The planned Monte Carlo design includes:

- DGPs: `dgp1`, `dgp2`, `dgp3`
- Sample sizes: `n = 250, 500, 1000`
- Control dimensions: `p = 200, 300, 500`
- Instrument strengths: `pi = 1.0, 0.5, 0.25, 0.10`
- Quantiles: `tau = 0.25, 0.50, 0.75`
- Replications: `R = 1000`
- DML cross-fitting folds: `K = 5`

## Current Status

Phase 1 architecture rewrite.

The repository currently contains a clean package skeleton only. Econometric logic for DGPs, estimators, inference, metrics, and simulation runners will be implemented in later phases.

Old prototype code has been archived in `archive_old/` so useful ideas can be recovered without mixing prototype modules into the new package.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Running Tests

```bash
pytest
```
