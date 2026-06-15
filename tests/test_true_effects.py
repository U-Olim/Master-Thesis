import ivqr_sim
from ivqr_sim.config import TAUS


def test_package_imports_work() -> None:
    assert ivqr_sim is not None
    assert TAUS == [0.25, 0.50, 0.75]
