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
    prepare_diagnostics_transposed_display,
    prepare_diagnostics_display,
    prepare_overall_transposed_display,
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
        [overall.assign(pi=pi) for pi in (0.1, 0.25, 0.5, 1.0)],
        ignore_index=True,
    ).to_csv(
        report_dir / "table_03_by_strength.csv", index=False
    )
    pd.concat(
        [
            overall.assign(n=n, p=p)
            for n, p in ((500, 200), (500, 500), (1000, 200), (1000, 500))
        ],
        ignore_index=True,
    ).to_csv(report_dir / "table_04_by_n_p.csv", index=False)
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
    assert display_estimator_label("Post-selection IVQR") == "Post-selection IVQR"
    assert (
        display_estimator_label("Post-selection IVQR", short=True)
        == "Post-selection IVQR"
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

    transposed = prepare_diagnostics_transposed_display(_metric_rows())
    dml_column = transposed["DML-style IVQR"]
    assert dml_column.loc[transposed["Diagnostic"].eq("Empty-region rate")].item() == "NA"
    assert (
        dml_column.loc[
            transposed["Diagnostic"].eq("Disconnected-region rate")
        ].item()
        == "NA"
    )


def test_overall_table_places_estimators_in_columns() -> None:
    display = prepare_overall_transposed_display(_metric_rows())
    assert display.columns.tolist() == [
        "Measure",
        "DML-style IVQR",
        "Oracle IVQR",
        "Post-selection IVQR",
    ]
    assert display["Measure"].tolist() == [
        "Coverage",
        "Bias",
        "MAE",
        "RMSE",
        "Estimate SD",
        "Mean CR length",
        "Median CR length",
        "Full-grid rate",
        "Unresolved rate",
    ]


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
    assert (output_dir / "figure_boundary_estimates.pdf").is_file()
    assert (output_dir / "figure_coverage_by_n_p.pdf").is_file()
    assert (output_dir / "table_overall_transposed.tex").is_file()
    overall_tex = (output_dir / "table_overall_transposed.tex").read_text(
        encoding="utf-8"
    )
    assert r"\textit{Note:}" not in overall_tex
    quantile_tex = (output_dir / "table_by_quantile_transposed.tex").read_text(
        encoding="utf-8"
    )
    assert quantile_tex.count(r"\addlinespace[0.35em]") == 2
    assert quantile_tex.count(r"\midrule") == 3
    strength_tex = (output_dir / "table_by_strength_transposed.tex").read_text(
        encoding="utf-8"
    )
    assert strength_tex.count(r"\addlinespace[0.35em]") == 3
    assert strength_tex.count(r"\midrule") == 4
    n_p_tex = (output_dir / "table_by_n_p_transposed.tex").read_text(
        encoding="utf-8"
    )
    assert n_p_tex.count(r"\addlinespace[0.35em]") == 3
    assert n_p_tex.count(r"\midrule") == 4
    displayed = pd.read_csv(output_dir / "table_overall.csv", keep_default_na=False)
    # The authoritative 0.91 is retained even though a one-row Bernoulli
    # recomputation could only produce zero or one.
    assert displayed.loc[0, "Measure"] == "Coverage"
    assert displayed.loc[0, "DML-style IVQR"] == "91.00%"
    dispersion = pd.read_csv(output_dir / "table_coverage_dispersion.csv")
    assert dispersion.columns.tolist() == [
        "Coverage-dispersion measure",
        *ESTIMATOR_DISPLAY_LABELS.values(),
    ]


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


def test_single_report_integrates_transposed_tables_and_figures() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = (repository / "documents/Full_Run_Results.qmd").read_text(
        encoding="utf-8"
    )
    assert not (repository / "documents/Full_Run_Results_Meeting.qmd").exists()
    assert not (repository / "documents/meeting-version.lua").exists()
    assert "toc: false" in report
    for asset in (
        "table_overall_transposed.tex",
        "table_coverage_dispersion_transposed.tex",
        "table_by_quantile_transposed.tex",
        "table_by_strength_transposed.tex",
        "table_by_n_p_transposed.tex",
        "figure_boundary_estimates.pdf",
        "figure_coverage_by_n_p.pdf",
    ):
        assert asset in report


def test_report_preserves_existing_estimator_descriptions() -> None:
    report = (
        Path(__file__).resolve().parents[1] / "documents/Full_Run_Results.qmd"
    ).read_text(encoding="utf-8")
    assert "# Estimators" in report
    assert "Oracle IVQR" in report
    assert "Post-selection IVQR" in report
    assert "mean-Lasso models" in report
    assert "union of variables" in report
    assert "model-selection uncertainty" in report
    assert "general post-selection-valid inference" in report
    assert "Mean-Lasso" not in report
    assert "PS-IVQR" not in report
    assert "Lasso Post-selection" not in report
    assert "DML-style IVQR" in report
