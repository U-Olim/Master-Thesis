from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulation.dml_output import (
    REQUIRED_DML_COLUMNS,
    clean_dml_results_csv,
    clean_dml_results_frame,
)
from reporting.summaries import aggregate_results


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
            "estimator": ["dml_ivqr", "dml_ivqr"],
            "alpha_hat": [1.1, 0.9],
            "alpha_true": [1.0, 1.0],
            "cr_lower": [0.5, np.nan],
            "cr_upper": [1.5, np.nan],
            "cr_length": [1.0, np.nan],
            "cr_covers_true": [True, False],
            "converged": [True, True],
            "runtime_seconds": [0.1, 0.2],
            "bias": [0.1, -0.1],
            "oracle_runtime_total_sec": [np.nan, np.nan],
            "ps_first_stage_r2": [np.nan, np.nan],
            "dml_quantile_solver": ["highs", "highs"],
            "message": ["ok", "empty confidence region"],
        }
    )


def test_clean_dml_output_exact_schema_and_no_extra_columns() -> None:
    cleaned, summary = clean_dml_results_frame(_wide_frame())

    assert tuple(cleaned.columns) == REQUIRED_DML_COLUMNS
    assert len(cleaned.columns) == 15
    assert summary.output_columns == 15
    assert not any("runtime" in column for column in cleaned.columns)
    assert set(cleaned.columns) == set(REQUIRED_DML_COLUMNS)


def test_clean_dml_output_preserves_rows_and_values() -> None:
    source = _wide_frame()
    cleaned, summary = clean_dml_results_frame(source)

    assert len(cleaned) == len(source) == summary.input_rows == summary.output_rows
    for column in (
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "rep",
        "seed",
        "estimator",
        "alpha_hat",
        "alpha_true",
        "cr_lower",
        "cr_upper",
        "cr_length",
        "converged",
    ):
        pd.testing.assert_series_equal(
            source[column], cleaned[column], check_dtype=False, check_names=False
        )


def test_clean_dml_output_preserves_missing_confidence_region() -> None:
    cleaned, summary = clean_dml_results_frame(_wide_frame())

    assert len(cleaned) == 2
    assert cleaned.loc[1, ["cr_lower", "cr_upper", "cr_length"]].isna().all()
    assert bool(cleaned.loc[1, "covered"]) is False
    assert summary.empty_confidence_regions == 1


def test_clean_dml_output_renames_covered_and_removes_runtime() -> None:
    cleaned, _ = clean_dml_results_frame(_wide_frame())

    assert "covered" in cleaned
    assert "cr_covers_true" not in cleaned
    assert cleaned["covered"].tolist() == [True, False]
    assert all("runtime" not in column.lower() for column in cleaned.columns)


def test_clean_dml_csv_writes_exact_header(tmp_path: Path) -> None:
    source = tmp_path / "wide.csv"
    output = tmp_path / "clean" / "dml.csv"
    _wide_frame().to_csv(source, index=False)

    clean_dml_results_csv(source, output)

    assert output.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        REQUIRED_DML_COLUMNS
    )
    assert tuple(pd.read_csv(output).columns) == REQUIRED_DML_COLUMNS


def test_clean_dml_output_reports_duplicates_without_deleting() -> None:
    source = pd.concat(
        [_wide_frame().iloc[[0]], _wide_frame().iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate dml_ivqr identifiers affect 2 rows"):
        clean_dml_results_frame(source)
    assert len(source) == 2


def test_clean_dml_output_rejects_impossible_coverage() -> None:
    source = _wide_frame()
    source.loc[0, ["cr_lower", "cr_upper"]] = [2.0, 3.0]

    with pytest.raises(ValueError, match="covered=True"):
        clean_dml_results_frame(source)


def test_clean_dml_output_rejects_missing_design_parameter() -> None:
    source = _wide_frame()
    source.loc[0, "seed"] = np.nan

    with pytest.raises(ValueError, match="seed must not be missing"):
        clean_dml_results_frame(source)


def test_clean_dml_output_remains_compatible_with_aggregation() -> None:
    cleaned, _ = clean_dml_results_frame(_wide_frame())

    summary = aggregate_results(cleaned, expected_replications=2)

    assert summary.loc[0, "bias"] == pytest.approx(0.0)
    assert summary.loc[0, "rmse"] == pytest.approx(0.1)
    assert summary.loc[0, "coverage"] == pytest.approx(0.5)
    assert summary.loc[0, "cr_empty_rate"] == pytest.approx(0.5)
    assert pd.isna(summary.loc[0, "mean_runtime_seconds"])
