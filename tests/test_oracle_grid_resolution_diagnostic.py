from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_oracle_grid_resolution.py"
SPEC = importlib.util.spec_from_file_location("diagnose_oracle_grid_resolution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def _cached_evaluations(
    alphas: np.ndarray, statistics: np.ndarray
) -> dict[float, tuple[object, float]]:
    return {
        float(alpha): (
            diagnostic.AlphaEvaluation(
                statistic=float(statistic),
                gamma_hat=np.array([0.0]),
                cov_gamma=np.array([[1.0]]),
                dim_z=1,
                converged=True,
                message="ok",
            ),
            0.01,
        )
        for alpha, statistic in zip(alphas, statistics, strict=True)
    }


def test_connected_components_are_preserved_and_serialized() -> None:
    alphas = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    statistics = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
    result = diagnostic._evaluate_grid(
        grid_variant="test",
        alphas=alphas,
        cached_evaluations=_cached_evaluations(alphas, statistics),
        alpha_true=0.0,
        direct_accepted=True,
    )

    assert result["number_of_connected_components"] == 2
    assert result["cr_components"].startswith("[[")
    assert result["cr_length"] < result["cr_upper"] - result["cr_lower"]


def test_boundary_interpolation_matches_linear_crossing() -> None:
    alphas = np.array([0.0, 1.0])
    statistics = np.array([1.0, 5.0])
    critical = 3.0

    assert diagnostic.interpolated_acceptance_at_alpha(
        alphas, statistics, 0.49, critical_value=critical
    )
    assert not diagnostic.interpolated_acceptance_at_alpha(
        alphas, statistics, 0.51, critical_value=critical
    )


def test_adaptive_refinement_stops_at_requested_transition_spacing() -> None:
    def statistic(alpha: float) -> float:
        return alpha

    grid = diagnostic.adaptive_refinement_grid(
        np.array([0.0, 0.2]),
        statistic,
        critical_value=0.07,
        max_spacing=0.025,
    )
    accepted = grid <= 0.07
    transition = np.flatnonzero(accepted[:-1] != accepted[1:])

    assert transition.size == 1
    index = int(transition[0])
    assert grid[index + 1] - grid[index] <= 0.025 + 1e-12
    assert len(grid) == 5


def test_exact_grid_point_coverage_consistency() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    accepted = np.array([False, True, False])

    exact, mismatch = diagnostic.exact_grid_consistency(
        alphas, accepted, alpha_true=1.0, covered=True
    )
    assert exact is True
    assert mismatch is False

    exact, mismatch = diagnostic.exact_grid_consistency(
        alphas,
        accepted,
        alpha_true=1.0,
        covered=True,
        direct_accepted=False,
    )
    assert exact is True
    assert mismatch is True
