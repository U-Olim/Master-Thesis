import numpy as np
import pytest
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp import Design, generate_data
from ivqr.alpha_grid import alpha_grid
from ivqr.ch_inverse import evaluate_alpha_ch_ivqr
from ivqr.confidence_regions import (
    FAILED_ALPHA_STATISTIC,
    invert_score_test,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
)
from ivqr.moments import quantile_score


def test_alpha_grid_size_and_bounds() -> None:
    grid = alpha_grid(-1.0, 3.0, 1.0)
    np.testing.assert_allclose(grid, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))


def test_confidence_region_inversion_on_known_vector() -> None:
    region = invert_score_test(
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([5.0, 1.0, 2.0, 6.0]),
        critical_value=3.0,
        alpha_true=1.0,
    )
    assert region.empty is False
    assert region.covers_true is True
    assert region.lower is not None
    assert region.upper is not None
    assert region.lower <= 0.0
    assert region.upper >= 1.0


def test_empty_confidence_region_diagnostics_are_stable() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    region = invert_score_test(
        alphas,
        np.array([5.0, 6.0, 7.0]),
        critical_value=1.0,
        alpha_true=0.0,
    )
    assert region.empty is True
    assert region.lower is None
    assert region.upper is None
    assert np.isnan(region.length)
    assert np.isnan(region.hull_length)
    assert region.covers_true is False

    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=np.array([False, False, False]),
        alpha_hat=None,
        failed_alpha_count=1,
    )
    assert np.isnan(diagnostics["cr_lower"])
    assert np.isnan(diagnostics["cr_upper"])
    assert np.isnan(diagnostics["cr_length"])
    assert diagnostics["cr_empty"] is True
    assert diagnostics["failed_alpha_rate"] == 1 / 3


def test_boundary_and_disconnected_region_diagnostics() -> None:
    alphas = np.array([-1.0, 0.0, 1.0, 2.0])
    accepted = np.array([True, False, True, True])
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=accepted,
        alpha_hat=-1.0,
        failed_alpha_count=0,
    )
    assert diagnostics["cr_hits_lower_boundary"] is True
    assert diagnostics["cr_hits_upper_boundary"] is True
    assert diagnostics["cr_hits_any_boundary"] is True
    assert diagnostics["cr_n_blocks"] == 2
    assert diagnostics["cr_disconnected"] is True


def test_quantile_score_basic_behavior() -> None:
    scores = quantile_score(np.array([-1.0, 0.0, 1.0]), tau=0.5)
    np.testing.assert_allclose(scores, np.array([-0.5, -0.5, 0.5]))


def test_ch_inverse_can_evaluate_tiny_grid() -> None:
    data = generate_data(Design("dgp1", 50, 4, 1.0, 0.5, rep=0, seed=123))
    evaluation = evaluate_alpha_ch_ivqr(
        y=data.y,
        d=data.d,
        x_controls=data.x[:, :2],
        z=data.z,
        alpha=1.0,
        tau=0.5,
        max_iter=100,
    )
    assert np.isfinite(evaluation.statistic)
    assert evaluation.usable


class _FakeQuantRegResult:
    def __init__(self, params: np.ndarray, covariance: np.ndarray) -> None:
        self.params = params
        self._covariance = covariance

    def cov_params(self) -> np.ndarray:
        return self._covariance


class _FakeQuantReg:
    result = _FakeQuantRegResult(np.array([0.0, 1.0]), np.eye(2))
    emit_iteration_warning = False

    def __init__(self, _y: np.ndarray, _design: np.ndarray) -> None:
        pass

    def fit(self, **_kwargs: object) -> _FakeQuantRegResult:
        if self.emit_iteration_warning:
            warnings.warn("iteration limit", IterationLimitWarning, stacklevel=2)
        return self.result


def _mock_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    params: np.ndarray,
    covariance: np.ndarray,
    warning: bool,
    policy: str | None = None,
    instruments: int = 1,
):
    _FakeQuantReg.result = _FakeQuantRegResult(params, covariance)
    _FakeQuantReg.emit_iteration_warning = warning
    monkeypatch.setattr("ivqr.ch_inverse.QuantReg", _FakeQuantReg)
    kwargs = {} if policy is None else {"iteration_warning_policy": policy}
    return evaluate_alpha_ch_ivqr(
        y=np.array([1.0, 2.0, 3.0]),
        d=np.array([0.2, 0.4, 0.6]),
        x_controls=np.empty((3, 0)),
        z=np.arange(3 * instruments, dtype=float).reshape(3, instruments),
        alpha=1.0,
        tau=0.5,
        **kwargs,
    )


@pytest.mark.parametrize("policy", ["reject", "use_if_valid"])
def test_no_warning_finite_fit_is_converged_and_usable(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 1.0]),
        covariance=np.eye(2),
        warning=False,
        policy=policy,
    )
    assert evaluation.converged
    assert evaluation.usable
    assert evaluation.warning_type is None
    assert evaluation.statistic == pytest.approx(1.0)


def test_valid_iteration_warning_is_used_by_production_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 1.0]),
        covariance=np.eye(2),
        warning=True,
    )
    assert not evaluation.converged
    assert evaluation.usable
    assert evaluation.warning_type == "iteration_limit"
    assert evaluation.statistic == pytest.approx(1.0)


def test_explicit_reject_reproduces_legacy_iteration_warning_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 1.0]),
        covariance=np.eye(2),
        warning=True,
        policy="reject",
    )
    assert not evaluation.converged
    assert not evaluation.usable
    assert evaluation.warning_type == "iteration_limit"
    assert np.isinf(evaluation.statistic)


def test_valid_iteration_warning_is_used_under_experimental_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 2.0]),
        covariance=np.diag([1.0, 2.0]),
        warning=True,
        policy="use_if_valid",
    )
    assert not evaluation.converged
    assert evaluation.usable
    assert evaluation.warning_type == "iteration_limit"
    assert evaluation.failure_reason is None
    assert evaluation.statistic == pytest.approx(2.0)


@pytest.mark.parametrize("policy", ["reject", "use_if_valid"])
def test_warning_with_nonfinite_parameters_is_unusable(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, np.nan]),
        covariance=np.eye(2),
        warning=True,
        policy=policy,
    )
    assert not evaluation.usable
    if policy == "use_if_valid":
        assert evaluation.failure_reason == "nonfinite_parameters"


@pytest.mark.parametrize(
    ("covariance", "reason"),
    [
        (np.array([[1.0, 0.0], [0.0, np.nan]]), "nonfinite_covariance"),
        (np.diag([1.0, -0.5]), "negative_instrument_variance"),
        (np.eye(1), "invalid_covariance_dimension"),
        (np.diag([1.0, 0.0]), "zero_instrument_variance"),
    ],
)
def test_warning_with_invalid_covariance_is_unusable(
    monkeypatch: pytest.MonkeyPatch, covariance: np.ndarray, reason: str
) -> None:
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 1.0]),
        covariance=covariance,
        warning=True,
        policy="use_if_valid",
    )
    assert not evaluation.usable
    assert evaluation.failure_reason == reason


def test_singular_multi_instrument_covariance_uses_finite_pseudoinverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    evaluation = _mock_evaluation(
        monkeypatch,
        params=np.array([0.0, 1.0, 1.0]),
        covariance=covariance,
        warning=True,
        policy="use_if_valid",
        instruments=2,
    )
    assert evaluation.usable
    assert evaluation.statistic == pytest.approx(1.0)
    assert evaluation.covariance_rank == 1
    assert evaluation.covariance_condition_number is not None
    assert evaluation.covariance_condition_number > 1e12


def test_grid_sanitization_uses_usability_not_warning_status() -> None:
    statistics, failures = sanitize_grid_statistics(
        np.array([2.0, 5.0, np.inf]), [True, True, False]
    )
    assert failures == 1
    np.testing.assert_allclose(statistics, np.array([2.0, 5.0, FAILED_ALPHA_STATISTIC]))
    assert statistics[0] <= 3.0
    assert statistics[1] > 3.0
