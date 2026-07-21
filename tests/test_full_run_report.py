from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.full_run_report import (
    ESTIMATOR_DISPLAY_LABELS,
    ESTIMATOR_ORDER,
    ReportInputError,
    compute_coverage_dispersion,
    display_estimator_label,
    format_percentage,
    generate_report_assets,
    order_estimators,
    prepare_diagnostics_display,
    validate_required_columns,
)


def _metric_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, label in enumerate(ESTIMATOR_ORDER):
        rows.append(
            {
                "estimator_label": label,
                "replications": 1,
                "resolved_replications": 1,
                "unresolved_replications": 0,
                "convergence_rate": 1.0,
                "bias": 0.01 * index,
                "median_bias": 0.0,
                "mae": 0.2 + index / 10,
                "rmse": 0.3 + index / 10,
                "estimate_sd": 0.4 + index / 10,
                "empirical_coverage": 0.91 + index / 100,
                "coverage_mcse": 0.01,
                "coverage_mc95_lower": 0.89 + index / 100,
                "coverage_mc95_upper": 0.93 + index / 100,
                "mean_cr_length": 2.0 + index / 10,
                "median_cr_length": 1.9 + index / 10,
                "full_grid_rate": 0.1 + index / 100,
                "empty_region_rate": np.nan if index == 0 else 0.0,
                "disconnected_region_rate": np.nan if index == 0 else 0.1,
                "unresolved_rate": 0.0,
                "boundary_estimate_rate": 0.02,
                "iteration_warning_rate": np.nan if index == 0 else 0.2,
                "rank_failure_rate": np.nan if index == 0 else 0.0,
                "refinement_limit_rate": np.nan if index == 0 else 0.0,
                "numerical_limit_rate": np.nan if index == 0 else 0.0,
                "mean_selected_controls": 12.0 if index == 2 else np.nan,
                "median_selected_controls": 12.0 if index == 2 else np.nan,
                "min_selected_controls": 10.0 if index == 2 else np.nan,
                "max_selected_controls": 14.0 if index == 2 else np.nan,
                "mean_retained_instruments": 1.0 if index == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_fixture(report_dir: Path) -> None:
    overall = _metric_rows()
    overall.to_csv(report_dir / "table_01_overall.csv", index=False)
    pd.concat(
        [overall.assign(tau=tau) for tau in (0.25, 0.5, 0.75)],
        ignore_index=True,
    ).to_csv(
        report_dir / "table_02_by_quantile.csv", index=False
    )
    pd.concat(
        [overall.assign(pi=pi) for pi in (0.1, 0.5, 1.0)],
        ignore_index=True,
    ).to_csv(
        report_dir / "table_03_by_strength.csv", index=False
    )
    overall.assign(n=[500, 500, 500], p=[200, 200, 200]).to_csv(
        report_dir / "table_04_by_n_p.csv", index=False
    )
    cells = overall.assign(
        dgp=["dgp1", "dgp1", "dgp1"],
        n=[500, 500, 500],
        p=[200, 200, 200],
        pi=[0.1, 0.5, 1.0],
        tau=[0.25, 0.5, 0.75],
    )
    cells.to_csv(report_dir / "table_05_by_design_cell.csv", index=False)
    cells.to_csv(report_dir / "table_06_worst_cells.csv", index=False)
    pd.DataFrame(
        {
            "estimator_label": ESTIMATOR_ORDER,
            "cr_status_standardized": [
                "observed_status_unavailable",
                "valid",
                "valid",
            ],
            "replications": [1, 1, 1],
            "resolved_replications": [1, 1, 1],
            "covered_resolved": [1, 1, 1],
            "uncovered_resolved": [0, 0, 0],
            "unresolved_replications": [0, 0, 0],
            "empirical_coverage": [0.91, 0.92, 0.93],
            "mean_cr_length": [2.0, 2.1, 2.2],
            "disconnected_region_rate": [np.nan, 0.1, 0.1],
            "iteration_warning_rate": [np.nan, 0.2, 0.2],
            "rank_failure_rate": [np.nan, 0.0, 0.0],
            "refinement_limit_rate": [np.nan, 0.0, 0.0],
            "numerical_limit_rate": [np.nan, 0.0, 0.0],
            "mean_selected_controls": [np.nan, np.nan, 12.0],
            "median_selected_controls": [np.nan, np.nan, 12.0],
            "mean_retained_instruments": [np.nan, np.nan, 1.0],
        }
    ).to_csv(report_dir / "table_07_diagnostics.csv", index=False)
    validation = {
        "reconciliation": {
            "all_row_totals_match": True,
            "resolved_totals_match": True,
            "overall_replications": 3,
            "overall_resolved_replications": 3,
        },
        "panel": {
            "DML-IVQR": {
                "rows": 1,
                "design_cells": 1,
                "replications_per_design_min": 1,
                "design_values": {
                    "dgp": ["dgp1"],
                    "n": [500],
                    "p": [200],
                    "pi": [0.1],
                    "tau": [0.25],
                },
            }
        },
    }
    (report_dir / "validation.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )


def test_required_columns_are_validated() -> None:
    with pytest.raises(ReportInputError, match="missing required columns"):
        validate_required_columns(pd.DataFrame({"a": [1]}), {"a", "b"}, "test.csv")


def test_estimator_order_is_deterministic() -> None:
    shuffled = pd.DataFrame(
        {"estimator_label": list(reversed(ESTIMATOR_ORDER)), "value": [1, 2, 3]}
    )
    assert order_estimators(shuffled)["estimator_label"].tolist() == ESTIMATOR_ORDER


def test_display_label_mapping_is_precise() -> None:
    assert display_estimator_label("DML-IVQR") == "DML-style IVQR"
    assert (
        display_estimator_label("Post-selection IVQR")
        == "Mean-Lasso Post-selection IVQR"
    )
    assert display_estimator_label("Post-selection IVQR", short=True) == (
        "Mean-Lasso PS-IVQR"
    )


def test_percentage_formatting_preserves_missingness() -> None:
    assert format_percentage(0.948983, digits=2) == "94.90%"
    assert format_percentage(np.nan) == "NA"


def test_dml_unavailable_diagnostics_do_not_become_zero() -> None:
    display = prepare_diagnostics_display(_metric_rows())
    dml = display.iloc[0]
    assert dml["Estimator"] == "DML-style IVQR"
    assert dml["Empty"] == "NA"
    assert dml["Disconnected"] == "NA"
    assert dml["Iteration warning"] == "NA"
    assert dml["Rank failure"] == "NA"
    assert dml["Refinement limit"] == "NA"


def test_small_fixture_generates_tables_and_figures_without_recomputing_coverage(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "validated"
    output_dir = tmp_path / "assets"
    report_dir.mkdir()
    _write_fixture(report_dir)

    outputs = generate_report_assets(report_dir, output_dir)

    assert output_dir / "figure_coverage_by_quantile.pdf" in outputs
    assert (output_dir / "figure_coverage_by_quantile.pdf").stat().st_size > 0
    assert (output_dir / "table_overall.tex").is_file()
    displayed = pd.read_csv(output_dir / "table_overall.csv", keep_default_na=False)
    # The authoritative 0.91 is retained even though a one-row Bernoulli
    # recomputation could only produce zero or one.
    assert displayed.loc[0, "Coverage"] == "91.00%"
    dispersion = pd.read_csv(output_dir / "table_coverage_dispersion.csv")
    assert dispersion["estimator_label"].tolist() == list(
        ESTIMATOR_DISPLAY_LABELS.values()
    )


def test_coverage_dispersion_uses_validated_cell_values_and_thresholds() -> None:
    coverage = {
        "DML-IVQR": [0.80, 0.90, 0.95, 1.00],
        "Oracle IVQR": [0.85, 0.925, 0.95, 0.975],
        "Post-selection IVQR": [0.70, 0.80, 0.90, 0.95],
    }
    rows: list[dict[str, object]] = []
    for estimator, values in coverage.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "estimator_label": estimator,
                    "dgp": f"dgp{index + 1}",
                    "n": 500,
                    "p": 200,
                    "pi": 0.5,
                    "tau": 0.5,
                    "empirical_coverage": value,
                }
            )

    result = compute_coverage_dispersion(pd.DataFrame(rows))

    assert result["estimator_label"].tolist() == list(
        ESTIMATOR_DISPLAY_LABELS.values()
    )
    dml = result.iloc[0]
    assert dml["p10_coverage"] == pytest.approx(np.quantile(coverage["DML-IVQR"], 0.1))
    assert dml["p25_coverage"] == pytest.approx(np.quantile(coverage["DML-IVQR"], 0.25))
    assert dml["share_below_90"] == pytest.approx(0.25)
    assert dml["share_below_92_5"] == pytest.approx(0.50)
    assert dml["share_below_95"] == pytest.approx(0.50)
    assert dml["share_at_or_above_95"] == pytest.approx(0.50)


def test_meeting_report_uses_shared_source_and_excludes_archival_cells() -> None:
    repository = Path(__file__).resolve().parents[1]
    meeting = (repository / "documents/Full_Run_Results_Meeting.qmd").read_text(
        encoding="utf-8"
    )
    common = (repository / "documents/Full_Run_Results.qmd").read_text(
        encoding="utf-8"
    )
    assert "{{< include Full_Run_Results.qmd >}}" in meeting
    assert "meeting-version: true" in meeting
    assert "table_design_cells.tex" in common
    assert "::: {.archival-only}" in common
    assert "table_design_cells.tex" not in meeting


def test_report_contains_implementation_specific_dml_limitation() -> None:
    report = (
        Path(__file__).resolve().parents[1] / "documents/Full_Run_Results.qmd"
    ).read_text(encoding="utf-8")
    assert "They do not constitute\na general validity result for DML-IVQR." in report
