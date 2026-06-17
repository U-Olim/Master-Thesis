# Consolidated tests for the thematic project structure.

import inference
from estimators.base import EstimationResult
from dgp.designs import Design
from inference import metrics


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None
