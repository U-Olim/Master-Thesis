from ivqr_sim import inference, metrics
from ivqr_sim.estimators.base import EstimationResult
from ivqr_sim.simulation.design import Design


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None
