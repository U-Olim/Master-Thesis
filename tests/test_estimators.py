import numpy as np

from dgp import Design, generate_data, get_oracle_control_indices
from estimators import (
    EstimationResult,
    estimate_dml_ivqr,
    estimate_full_control_ivqr,
    estimate_oracle_ivqr,
    estimate_post_selection_ivqr,
)


def _tiny_data():
    return generate_data(Design("dgp1", 80, 10, 1.0, 0.5, rep=0, seed=321))


def test_estimators_return_estimation_results_with_expected_names() -> None:
    data = _tiny_data()
    alphas = np.linspace(-1.0, 3.0, 5)
    results = [
        estimate_oracle_ivqr(
            data,
            tau=0.5,
            alphas=alphas,
            oracle_indices=get_oracle_control_indices("dgp1", data.x.shape[1]),
            max_iter=100,
        ),
        estimate_post_selection_ivqr(
            data,
            tau=0.5,
            alphas=alphas,
            selection_cv=2,
            quantreg_max_iter=100,
        ),
        estimate_full_control_ivqr(data, tau=0.5, alphas=alphas, max_iter=100),
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=2),
    ]
    assert all(isinstance(result, EstimationResult) for result in results)
    assert [result.estimator for result in results] == [
        "oracle",
        "post_selection_ivqr",
        "full_control_ivqr",
        "dml_ivqr",
    ]
