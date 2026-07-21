from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import RAW_MANIFEST_PATH, RAW_RESULT_FILES, verify_raw_manifest
from analysis.r500_audit import artifact_metadata
from analysis.r500_phase2 import (
    WARNING_DEFINITIONS,
    classify_estimators,
    classify_exception,
    coverage_uncertainty,
    exception_diagnostics,
    paired_comparisons,
    warning_membership,
    warning_summaries,
    wilson_interval,
)


def _frame(estimator: str) -> pd.DataFrame:
    label = {
        "oracle": "oracle",
        "post_selection": "post_selection_ivqr",
        "dml": "dml_ivqr",
    }[estimator]
    frame = pd.DataFrame(
        {
            "dgp": ["dgp1"] * 3,
            "n": [500] * 3,
            "p": [200] * 3,
            "pi": [0.1] * 3,
            "tau": [0.5] * 3,
            "rep": [0, 1, 2],
            "seed": [10, 11, 12],
            "estimator": [label] * 3,
            "alpha_true": [1.0] * 3,
            "alpha_hat": [1.0, 1.5, 0.0],
            "covered": [True, True, False],
            "cr_lower": [0.0, 0.0, 1.2],
            "cr_upper": [2.0, 2.0, 1.8],
            "cr_length": [2.0, 2.0, 0.6],
            "converged": [True] * 3,
        }
    )
    if estimator in {"oracle", "post_selection"}:
        frame = frame.assign(
            cr_components=["[[0.0,2.0]]", "[[0.0,2.0]]", "[[1.2,1.8]]"],
            cr_n_blocks=[1, 1, 1],
            cr_disconnected=[False, False, False],
            cr_status=["valid", "valid", "valid"],
            cr_is_numerically_resolved=[True, True, True],
            cr_unresolved_count=[0, 0, 0],
            cr_unresolved_alphas=["[]", "[]", "[]"],
            iteration_warning_evaluations=[1, 2, 0],
            rank_deficient_covariance_failures=[0, 0, 0],
            midpoint_unresolved_barriers=[1, 0, 0],
            number_of_unresolved_refinement_barriers=[1, 0, 0],
            midpoint_probe_limit_hit=[False] * 3,
            refinement_limit_hit=[False] * 3,
            max_alpha_evaluations_hit=[False] * 3,
        )
    return frame


def _frames() -> dict[str, pd.DataFrame]:
    frames = {name: _frame(name) for name in ("oracle", "post_selection", "dml")}
    frames["post_selection"]["covered"] = [True, False, False]
    frames["dml"]["covered"] = [False, True, False]
    return frames


def test_warning_membership_is_multilabel_and_counts_events() -> None:
    membership, metadata = warning_membership(_frame("oracle"))
    first = membership[membership["rep"].eq(0)]
    assert set(first["warning_category"]) == {
        "iteration_warning",
        "midpoint_unresolved_barrier",
        "refinement_unresolved_barrier",
    }
    iteration = membership[membership["warning_category"].eq("iteration_warning")]
    assert iteration["warning_events"].sum() == 3
    assert metadata["textual_warning_reason_available"] is False


def test_warning_reason_fallback_does_not_fabricate_categories() -> None:
    membership, metadata = warning_membership(_frame("dml"))
    assert membership.empty
    assert set(metadata["unavailable_categories"]) == set(WARNING_DEFINITIONS)


def test_warning_summary_reports_affected_and_unaffected_denominators() -> None:
    summary, scenarios, taxonomy = warning_summaries(_frames())
    row = summary[
        summary["estimator"].eq("oracle")
        & summary["warning_category"].eq("iteration_warning")
    ].iloc[0]
    assert row["warning_count"] == 2
    assert row["warning_event_count"] == 3
    assert row["coverage_affected_denominator"] == 2
    assert row["coverage_without_warning_denominator"] == 1
    assert len(scenarios) == 14
    assert "limitation" in taxonomy


def test_exception_classification_is_conservative() -> None:
    empty = _frame("oracle").iloc[0].copy()
    empty["cr_status"] = "empty_valid"
    empty["cr_components"] = "[]"
    empty[["cr_lower", "cr_upper", "cr_length"]] = np.nan
    assert classify_exception(empty, "oracle")[0] == "complete_rejection_across_evaluated_grid"

    unresolved = _frame("post_selection").iloc[0].copy()
    unresolved["cr_is_numerically_resolved"] = False
    unresolved["cr_unresolved_count"] = 1
    assert classify_exception(unresolved, "post_selection")[0] == "numerical_non_resolution"

    missing = _frame("dml").iloc[0].copy()
    missing[["cr_lower", "cr_upper", "cr_length"]] = np.nan
    assert classify_exception(missing, "dml")[0] == "missing_legacy_geometry"


def test_exception_diagnostics_distinguish_empty_unresolved_and_dml_missing() -> None:
    frames = _frames()
    frames["oracle"].loc[0, ["cr_lower", "cr_upper", "cr_length"]] = np.nan
    frames["oracle"].loc[0, "cr_components"] = "[]"
    frames["oracle"].loc[0, "cr_status"] = "empty_valid"
    frames["post_selection"].loc[1, "cr_is_numerically_resolved"] = False
    frames["post_selection"].loc[1, "cr_unresolved_count"] = 1
    frames["post_selection"].loc[1, "cr_status"] = "partially_unresolved"
    frames["dml"].loc[2, ["cr_lower", "cr_upper", "cr_length"]] = np.nan
    rows, summary = exception_diagnostics(frames)
    assert rows["exception_type"].tolist() == [
        "missing_legacy_geometry",
        "empty_cr",
        "unresolved_cr",
    ]
    assert not rows["invalid_geometry"].any()
    assert summary["exception_count"].sum() == 3


def test_paired_comparisons_validate_keys_seeds_and_orientation() -> None:
    overall, scenarios, discordance = paired_comparisons(_frames())
    row = overall[
        overall["estimator_a"].eq("oracle")
        & overall["estimator_b"].eq("post_selection")
        & overall["metric"].eq("coverage")
    ].iloc[0]
    assert row["valid_paired_denominator"] == 3
    assert row["both_cover"] == 1
    assert row["only_a_covers"] == 1
    assert row["only_b_covers"] == 0
    assert row["neither_covers"] == 1
    assert row["mean_paired_difference"] == pytest.approx(1 / 3)
    assert len(scenarios) == 12
    assert len(discordance) == 6

    frames = _frames()
    frames["dml"].loc[0, "seed"] = 99
    with pytest.raises(ValueError, match="Seed conflict"):
        paired_comparisons(frames)


def test_paired_comparisons_fail_on_duplicates_and_key_mismatch() -> None:
    frames = _frames()
    frames["oracle"] = pd.concat(
        [frames["oracle"], frames["oracle"].iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate natural keys"):
        paired_comparisons(frames)
    frames = _frames()
    frames["dml"].loc[0, "rep"] = 9
    with pytest.raises(ValueError, match="Paired-key mismatch"):
        paired_comparisons(frames)


def test_paired_metric_denominator_excludes_nonfinite_values() -> None:
    frames = _frames()
    frames["post_selection"].loc[0, "cr_length"] = np.nan
    overall, _, _ = paired_comparisons(frames)
    row = overall[
        overall["estimator_a"].eq("oracle")
        & overall["estimator_b"].eq("post_selection")
        & overall["metric"].eq("cr_length")
    ].iloc[0]
    assert row["valid_paired_denominator"] == 2


def test_wilson_interval_and_coverage_uncertainty() -> None:
    lower, upper = wilson_interval(475, 500)
    assert lower == pytest.approx(0.9273, abs=1e-4)
    assert upper == pytest.approx(0.9659, abs=1e-4)
    scenarios = pd.DataFrame(
        {
            "estimator": ["oracle"], "dgp": ["dgp1"], "n": [500], "p": [200],
            "pi": [0.1], "tau": [0.5], "conditional_coverage": [0.95],
            "coverage_denominator": [500], "coverage_mcse": [0.01],
            "coverage_mc95_lower": [0.93], "coverage_mc95_upper": [0.97],
        }
    )
    result = coverage_uncertainty(scenarios).iloc[0]
    assert result["coverage_successes"] == 475
    assert result["wilson_includes_nominal"]


def test_formal_classification_thresholds_and_diagnostic_downgrade() -> None:
    summary = pd.DataFrame(
        {
            "estimator": ["oracle", "post_selection", "dml"],
            "conditional_coverage": [0.95, 0.92, 0.95],
            "coverage_denominator": [1000, 1000, 1000],
            "unresolved_rate": [0.0, 0.0, np.nan],
            "empty_valid_rate": [0.0, 0.0, np.nan],
            "rmse": [1.0, 1.05, 1.0],
            "mean_cr_length": [2.0, 2.0, 2.0],
        }
    )
    scenarios = pd.DataFrame(
        {
            "estimator": ["oracle", "post_selection", "dml"],
            "coverage_gap": [0.0, -0.03, 0.0],
        }
    )
    result, rules = classify_estimators(summary, scenarios, _frames())
    classes = result.set_index("estimator")["classification"].to_dict()
    assert classes == {
        "dml": "acceptable with caveats",
        "oracle": "strong",
        "post_selection": "concerning",
    }
    assert result.set_index("estimator").loc["dml", "diagnostic_confidence"] == "limited"
    assert rules["no_composite_score"] is True


def test_outputs_do_not_depend_on_input_order_and_csv_bytes_are_stable(tmp_path: Path) -> None:
    frames = _frames()
    first, _, _ = paired_comparisons(frames)
    shuffled = {
        name: frame.sample(frac=1, random_state=2).reset_index(drop=True)
        for name, frame in frames.items()
    }
    second, _, _ = paired_comparisons(shuffled)
    pd.testing.assert_frame_equal(first, second)
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for path, frame in zip(paths, (first, second), strict=True):
        frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_real_source_artifacts_are_not_mutated() -> None:
    paths = [*RAW_RESULT_FILES.values(), RAW_MANIFEST_PATH]
    before = {Path(path): artifact_metadata(path) for path in paths}
    verify_raw_manifest()
    after = {Path(path): artifact_metadata(path) for path in paths}
    assert after == before
