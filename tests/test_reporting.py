from pathlib import Path

import pandas as pd

from reporting.summaries import aggregate_results
from reporting.tables import ESTIMATOR_ORDER, write_tables


def _raw() -> pd.DataFrame:
    rows = []
    for estimator in ("oracle", "post_selection_ivqr", "full_control_ivqr", "dml_ivqr"):
        rows.append(
            {
                "dgp": "dgp1",
                "n": 50,
                "p": 4,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": estimator,
                "alpha_hat": 1.0,
                "alpha_true": 1.0,
                "failed": False,
                "converged": True,
                "cr_length": 1.0,
                "cr_hull_length": 1.0,
                "cr_covers_true": True,
                "cr_empty": False,
                "cr_disconnected": False,
                "cr_hits_any_boundary": False,
                "alpha_hat_at_any_boundary": False,
                "failed_alpha_count": 0,
                "failed_alpha_rate": 0.0,
                "selected_controls": 2,
                "critical_value_multiplier": 1.0,
                "critical_value_adjusted": 3.84,
                "runtime_seconds": 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_summary_and_tables_work_on_tiny_frame(tmp_path: Path) -> None:
    assert ESTIMATOR_ORDER == (
        "oracle",
        "post_selection_ivqr",
        "full_control_ivqr",
        "dml_ivqr",
    )
    summary = aggregate_results(_raw(), expected_replications=1)
    written = write_tables(summary, tmp_path)
    assert (tmp_path / "comparison_table.csv").exists()
    assert "comparison" in written
    assert set(summary["estimator"]) == set(ESTIMATOR_ORDER)
