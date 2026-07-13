from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reporting.summaries import aggregate_results
from simulation.oracle_output import (
    REQUIRED_ORACLE_COLUMNS,
    clean_oracle_results_csv,
    clean_oracle_results_frame,
)


def _wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1"],
            "n": [100, 100],
            "p": [10, 10],
            "pi": [0.5, 0.5],
            "tau": [0.5, 0.5],
            "rep": [0, 1],
            "seed": [123, 124],
            "estimator": ["oracle", "oracle"],
            "alpha_hat": [1.1, 0.9],
            "alpha_true": [1.0, 1.0],
            "cr_lower": [0.5, np.nan],
            "cr_upper": [1.5, np.nan],
            "cr_length": [1.0, np.nan],
            "cr_covers_true": [True, False],
            "converged": [True, True],
            "runtime_seconds": [0.1, 0.2],
            "oracle_runtime_total_sec": [0.1, 0.2],
            "bias": [0.1, -0.1],
            "selected_controls": [5, 5],
            "ps_first_stage_r2": [np.nan, np.nan],
            "dml_quantile_solver": [np.nan, np.nan],
            "critical_value": [3.84, 3.84],
            "message": ["ok", "empty confidence region"],
        }
    )


def test_clean_oracle_exact_schema_and_no_extra_columns() -> None:
    cleaned, summary = clean_oracle_results_frame(_wide_frame())

    assert tuple(cleaned.columns) == REQUIRED_ORACLE_COLUMNS
    assert len(cleaned.columns) == 15 == summary.output_columns
    assert set(cleaned.columns) == set(REQUIRED_ORACLE_COLUMNS)
    assert not any("runtime" in column for column in cleaned.columns)


def test_clean_oracle_preserves_rows_and_values() -> None:
    source = _wide_frame()
    cleaned, summary = clean_oracle_results_frame(source)

    assert len(cleaned) == len(source) == summary.input_rows == summary.output_rows
    for column in REQUIRED_ORACLE_COLUMNS:
        source_column = "cr_covers_true" if column == "covered" else column
        pd.testing.assert_series_equal(
            source[source_column],
            cleaned[column],
            check_dtype=False,
            check_names=False,
        )


def test_clean_oracle_preserves_missing_confidence_region() -> None:
    cleaned, summary = clean_oracle_results_frame(_wide_frame())

    assert len(cleaned) == 2
    assert cleaned.loc[1, ["cr_lower", "cr_upper", "cr_length"]].isna().all()
    assert bool(cleaned.loc[1, "covered"]) is False
    assert summary.empty_confidence_regions == 1


def test_clean_oracle_renames_covered_and_removes_diagnostics() -> None:
    cleaned, _ = clean_oracle_results_frame(_wide_frame())

    assert cleaned["covered"].tolist() == [True, False]
    assert "cr_covers_true" not in cleaned
    assert "oracle_runtime_total_sec" not in cleaned
    assert "selected_controls" not in cleaned
    assert "critical_value" not in cleaned


def test_clean_oracle_csv_writes_exact_header(tmp_path: Path) -> None:
    source = tmp_path / "wide.csv"
    output = tmp_path / "clean" / "oracle.csv"
    _wide_frame().to_csv(source, index=False)

    clean_oracle_results_csv(source, output)

    assert output.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        REQUIRED_ORACLE_COLUMNS
    )
    assert tuple(pd.read_csv(output).columns) == REQUIRED_ORACLE_COLUMNS


def test_clean_oracle_rejects_duplicate_identifiers_without_deleting() -> None:
    source = pd.concat(
        [_wide_frame().iloc[[0]], _wide_frame().iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate oracle identifiers"):
        clean_oracle_results_frame(source)
    assert len(source) == 2


def test_clean_oracle_rejects_wrong_estimator() -> None:
    source = _wide_frame()
    source.loc[0, "estimator"] = "oracle_ivqr"

    with pytest.raises(ValueError, match="other than oracle"):
        clean_oracle_results_frame(source)


def test_clean_oracle_remains_compatible_with_aggregation() -> None:
    cleaned, _ = clean_oracle_results_frame(_wide_frame())

    summary = aggregate_results(cleaned, expected_replications=2)

    assert summary.loc[0, "bias"] == pytest.approx(0.0)
    assert summary.loc[0, "rmse"] == pytest.approx(0.1)
    assert summary.loc[0, "coverage"] == pytest.approx(0.5)
    assert summary.loc[0, "cr_empty_rate"] == pytest.approx(0.5)
    assert pd.isna(summary.loc[0, "mean_runtime_seconds"])
