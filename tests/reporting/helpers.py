import pandas as pd


def raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1", "dgp1", "dgp1"],
            "n": [80, 80, 80, 80],
            "p": [5, 5, 5, 5],
            "pi": [1.0, 1.0, 1.0, 1.0],
            "tau": [0.5, 0.5, 0.5, 0.5],
            "rep": [0, 1, 0, 1],
            "estimator": [
                "dml_ivqr",
                "dml_ivqr",
                "post_selection_ivqr",
                "post_selection_ivqr",
            ],
            "alpha_hat": [1.1, 0.9, 1.2, None],
            "alpha_true": [1.0, 1.0, 1.0, 1.0],
            "failed": [False, False, False, True],
            "converged": [True, True, True, False],
            "cr_length": [1.0, 2.0, 1.5, None],
            "cr_covers_true": [True, True, False, None],
            "cr_empty": [False, False, False, True],
            "cr_disconnected": [False, False, True, None],
            "runtime_seconds": [0.1, 0.2, 0.3, 0.4],
            "failed_alpha_count": [0, 0, 0, 1],
            "selected_controls": [None, None, 3, 4],
        }
    )

def r10_style_raw_results() -> pd.DataFrame:
    rows = []
    for rep in range(2):
        rows.extend(
            [
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "oracle",
                    "alpha_hat": 1.0 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 1.0,
                    "cr_covers_true": rep == 0,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.1,
                    "failed_alpha_count": 0,
                    "selected_controls": 10,
                },
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "post_selection_ivqr",
                    "alpha_hat": 0.9 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 1.5,
                    "cr_covers_true": True,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.2,
                    "failed_alpha_count": 0,
                    "selected_controls": 20,
                },
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "dml_ivqr",
                    "alpha_hat": 1.1 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 2.0,
                    "cr_covers_true": True,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.3,
                    "failed_alpha_count": 0,
                    "selected_controls": None,
                },
            ]
        )
    return pd.DataFrame(rows)

def summary() -> pd.DataFrame:
    rows = []
    for dgp, n, p, pi, tau in [
        ("dgp1", 80, 5, 1.0, 0.5),
        ("dgp1", 100, 10, 0.5, 0.75),
    ]:
        rows.extend(
            [
                {
                    "dgp": dgp,
                    "n": n,
                    "p": p,
                    "pi": pi,
                    "tau": tau,
                    "estimator": "dml_ivqr",
                    "bias": 0.123456,
                    "median_bias": 0.1,
                    "rmse": 0.23456,
                    "mae": 0.2,
                    "coverage": 0.95,
                    "avg_cr_length": 1.23456,
                    "avg_cr_length_valid_only": 1.5,
                    "failure_rate": 0.0,
                    "non_convergence_rate": 0.0,
                    "cr_empty_rate": 0.0,
                    "cr_disconnected_rate": 0.0,
                    "mean_runtime_seconds": 0.45678,
                    "replications": 2,
                    "valid_estimates": 2,
                    "expected_replications": 2,
                    "observed_replications": 2,
                    "completion_rate": 1.0,
                    "boundary_rate": 0.0,
                    "mean_failed_alpha_count": 0.0,
                    "mean_selected_controls": 3.0,
                },
                {
                    "dgp": dgp,
                    "n": n,
                    "p": p,
                    "pi": pi,
                    "tau": tau,
                    "estimator": "post_selection_ivqr",
                    "bias": -0.2,
                    "median_bias": -0.2,
                    "rmse": 0.3,
                    "mae": 0.25,
                    "coverage": 0.9,
                    "avg_cr_length": 1.5,
                    "avg_cr_length_valid_only": 2.0,
                    "failure_rate": 0.1,
                    "non_convergence_rate": 0.1,
                    "cr_empty_rate": 0.05,
                    "cr_disconnected_rate": 0.0,
                    "mean_runtime_seconds": 0.8,
                    "replications": 2,
                    "valid_estimates": 1,
                    "expected_replications": 2,
                    "observed_replications": 2,
                    "completion_rate": 1.0,
                    "boundary_rate": 0.0,
                    "mean_failed_alpha_count": 1.0,
                    "mean_selected_controls": 4.0,
                },
            ]
        )
    return pd.DataFrame(rows)

def row(summary: pd.DataFrame, estimator: str) -> pd.Series:
    return summary.loc[summary["estimator"] == estimator].iloc[0]
