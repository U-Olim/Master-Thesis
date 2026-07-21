from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import RAW_MANIFEST_PATH, RAW_RESULT_FILES, verify_raw_manifest
from analysis.r500_audit import (
    NATURAL_KEY,
    artifact_metadata,
    harmonize_frames,
    monte_carlo_interval,
    summarize_estimator,
    summarize_scenarios,
    suspicious_patterns,
    validate_alignment,
    validate_cr_components_frame,
    validate_result_values,
    validate_structure,
    worst_scenarios,
)


LABELS = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}
SMALL_DESIGN = {
    "dgp": ("dgp1",),
    "n": (500,),
    "p": (200,),
    "pi": (0.1,),
    "tau": (0.5,),
}


def _frame(estimator: str, *, resolved: tuple[bool, bool] = (True, True)) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1"],
            "n": [500, 500],
            "p": [200, 200],
            "pi": [0.1, 0.1],
            "tau": [0.5, 0.5],
            "rep": [0, 1],
            "seed": [10, 11],
            "estimator": [LABELS[estimator]] * 2,
            "alpha_true": [1.0, 1.0],
            "alpha_hat": [1.5, 0.5],
            "covered": [True, False],
            "cr_lower": [0.0, 1.2],
            "cr_upper": [2.0, 1.8],
            "cr_length": [2.0, 0.6],
            "converged": [True, True],
        }
    )
    if estimator in {"oracle", "post_selection"}:
        frame = frame.assign(
            cr_status=["valid", "valid"],
            cr_n_blocks=[1, 1],
            cr_disconnected=[False, False],
            cr_is_numerically_resolved=list(resolved),
            cr_unresolved_count=[0, int(not resolved[1])],
            iteration_warning_evaluations=[0, 2],
        )
    if estimator == "post_selection":
        frame = frame.assign(
            n_selected_controls=[3, 5],
            selection_lasso_multiplier=[1.0, 1.0],
            n_retained_instruments=[1, 1],
            selection_method=["post_lasso", "post_lasso"],
        )
    return frame


def _frames() -> dict[str, pd.DataFrame]:
    return {name: _frame(name) for name in ("oracle", "post_selection", "dml")}


def test_structure_accepts_complete_small_design_and_reports_sorting() -> None:
    result = validate_structure(
        _frame("oracle"),
        "oracle",
        expected_replications=2,
        expected_values=SMALL_DESIGN,
    )
    assert result["rows"] == 2
    assert result["design_cells"] == 1
    assert result["rows_per_replication"] == [1]
    assert result["naturally_sorted"] is True


def test_structure_rejects_missing_cell_and_duplicate_natural_key() -> None:
    frame = _frame("oracle")
    with pytest.raises(ValueError, match="incomplete design cells"):
        validate_structure(
            frame.iloc[:1],
            "oracle",
            expected_replications=2,
            expected_values=SMALL_DESIGN,
        )
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate natural keys"):
        validate_structure(
            duplicate,
            "oracle",
            expected_replications=2,
            expected_values=SMALL_DESIGN,
        )


def test_alignment_rejects_seed_and_key_mismatches() -> None:
    frames = _frames()
    frames["dml"].loc[0, "seed"] = 99
    with pytest.raises(ValueError, match="Seed mismatch"):
        validate_alignment(frames)

    frames = _frames()
    frames["dml"].loc[0, "rep"] = 8
    with pytest.raises(ValueError, match="natural-key mismatch"):
        validate_alignment(frames)


def test_harmonized_metrics_use_explicit_coverage_denominators() -> None:
    frames = _frames()
    frames["post_selection"] = _frame(
        "post_selection", resolved=(True, False)
    )
    harmonized = harmonize_frames(frames)
    summary = summarize_estimator(harmonized).set_index("estimator")
    scenarios = summarize_scenarios(harmonized)

    assert summary.loc["oracle", "bias"] == pytest.approx(0.0)
    assert summary.loc["oracle", "rmse"] == pytest.approx(0.5)
    assert summary.loc["oracle", "coverage_denominator"] == 2
    assert summary.loc["post_selection", "coverage_denominator"] == 1
    assert summary.loc["post_selection", "unresolved_rows"] == 1
    assert summary.loc["post_selection", "conditional_excluded_rows"] == 1
    assert np.isnan(summary.loc["dml", "unresolved_rows"])
    assert np.isnan(summary.loc["dml", "unresolved_rate"])
    assert len(scenarios) == 3


def test_post_selection_multiplier_and_selection_metrics_are_reported() -> None:
    summary = summarize_estimator(harmonize_frames(_frames())).set_index("estimator")
    row = summary.loc["post_selection"]
    assert row["selection_multiplier_values"] == "1:2"
    assert row["mean_selected_controls"] == pytest.approx(4.0)
    assert row["mean_retained_instruments"] == pytest.approx(1.0)


def test_monte_carlo_interval_uses_effective_denominator() -> None:
    se, lower, upper = monte_carlo_interval(0.9, 100)
    assert se == pytest.approx(0.03)
    assert lower == pytest.approx(0.8412)
    assert upper == pytest.approx(0.9588)
    assert all(np.isnan(value) for value in monte_carlo_interval(np.nan, 0))


def test_component_validation_uses_disconnected_components_for_coverage() -> None:
    frame = _frame("oracle").iloc[[0]].copy()
    frame["alpha_true"] = 0.0
    frame["covered"] = False
    frame["cr_lower"] = -1.0
    frame["cr_upper"] = 1.0
    frame["cr_length"] = 1.0
    frame["cr_components"] = '[[ -1.0, -0.5 ], [ 0.5, 1.0 ]]'
    frame["cr_n_blocks"] = 2
    frame["cr_disconnected"] = True
    result = validate_cr_components_frame(frame, "oracle")
    assert result["rows_checked"] == 1

    frame["covered"] = True
    with pytest.raises(ValueError, match="component-based coverage mismatches"):
        validate_cr_components_frame(frame, "oracle")


def test_common_value_validation_rejects_negative_lengths_and_reversed_bounds() -> None:
    frame = _frame("dml")
    frame.loc[0, "cr_length"] = -1.0
    with pytest.raises(ValueError, match="negative CR lengths"):
        validate_result_values(frame, "dml")
    frame = _frame("dml")
    frame.loc[0, ["cr_lower", "cr_upper"]] = [2.0, 0.0]
    with pytest.raises(ValueError, match="reversed CR bounds"):
        validate_result_values(frame, "dml")


def test_harmonization_and_scenario_outputs_are_deterministically_sorted() -> None:
    frames = {
        name: frame.iloc[::-1].reset_index(drop=True)
        for name, frame in _frames().items()
    }
    harmonized = harmonize_frames(frames)
    expected = harmonized.sort_values(
        ["estimator", *NATURAL_KEY], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(harmonized, expected)


def test_worst_scenario_ranking_is_metric_labelled_and_stable() -> None:
    scenarios = pd.DataFrame(
        {
            "estimator": ["oracle", "oracle"],
            "dgp": ["dgp2", "dgp1"],
            "n": [500, 500],
            "p": [200, 200],
            "pi": [0.1, 0.1],
            "tau": [0.5, 0.5],
            "coverage_gap": [-0.1, -0.1],
            "rmse": [2.0, 1.0],
            "mean_cr_length": [3.0, 2.0],
            "nonconvergence_rate": [0.0, 0.1],
            "unresolved_rate": [0.0, 0.0],
            "coverage_denominator": [500, 500],
            "coverage_mcse": [0.01, 0.01],
        }
    )
    ranked = worst_scenarios(scenarios, top_n=2)
    coverage = ranked[ranked.metric == "coverage_gap"]
    assert coverage.iloc[0].dgp == "dgp1"
    assert coverage["rank"].tolist() == [1, 2]


def test_suspicious_patterns_flag_values_without_fabricating_dml_diagnostics() -> None:
    frames = _frames()
    frames["dml"].loc[0, "cr_length"] = -1.0
    scenarios = summarize_scenarios(harmonize_frames(_frames()))
    diagnostics = suspicious_patterns(frames, scenarios).set_index(
        ["estimator", "check"]
    )
    assert diagnostics.loc[("dml", "negative_cr_length"), "count"] == 1
    assert diagnostics.loc[
        ("dml", "connected_coverage_inconsistency"), "count"
    ] == 0


def test_real_manifest_integration_is_read_only() -> None:
    paths = [*RAW_RESULT_FILES.values(), RAW_MANIFEST_PATH]
    before = {Path(path): artifact_metadata(path) for path in paths}
    verify_raw_manifest()
    after = {Path(path): artifact_metadata(path) for path in paths}
    assert after == before
    assert [before[Path(path)]["column_count"] for path in paths] == [43, 52, 15, None]
    assert [before[Path(path)]["row_count"] for path in paths[:3]] == [72000] * 3
