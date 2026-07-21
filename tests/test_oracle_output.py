from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.data import load_oracle_results
from analysis.tables import summarize_performance
from dgp import Design
from simulation.dml_output import REQUIRED_DML_COLUMNS
from simulation.oracle_output import (
    ORACLE_OUTPUT_COLUMNS,
    clean_oracle_results_frame,
    serialize_oracle_result,
)
from simulation.post_selection_output import REQUIRED_POST_SELECTION_COLUMNS
from simulation.runner import (
    filter_completed_designs,
    make_simulation_grid,
    run_simulation_design,
)


def _expanded_row(rep: int = 0) -> dict[str, object]:
    return {
        "dgp": "dgp1",
        "n": 100,
        "p": 10,
        "pi": 0.5,
        "tau": 0.5,
        "rep": rep,
        "seed": 123 + rep,
        "result_schema_version": 4,
        "estimator": "oracle",
        "alpha_true": 1.0,
        "alpha_hat": 1.125,
        "cr_covers_true": True,
        "cr_length": 0.75,
        "cr_status": "valid",
        "cr_n_blocks": 2,
        "iteration_warning_evaluations": 3,
        "cr_lower": 0.5,
        "cr_upper": 1.5,
        "cr_disconnected": True,
        "cr_components": "[[0.5,0.75],[1.0,1.5]]",
        "converged": True,
        "cr_is_numerically_resolved": True,
        "cr_unresolved_count": 0,
        "final_alpha_evaluations": 31,
        "refinement_depth_reached": 4,
        "number_of_refined_intervals": 5,
        "minimum_final_grid_spacing": 0.01,
        "median_final_grid_spacing": 0.05,
        "grid_strategy": "adaptive",
        "refinement_limit_hit": False,
    }


def test_serialized_oracle_result_has_exact_output_schema() -> None:
    serialized = serialize_oracle_result(_expanded_row())
    assert list(serialized) == list(ORACLE_OUTPUT_COLUMNS)
    assert len(serialized) == 26


def test_oracle_projection_preserves_values_and_types() -> None:
    source = _expanded_row()
    serialized = serialize_oracle_result(source)
    for column in ORACLE_OUTPUT_COLUMNS:
        source_column = "cr_covers_true" if column == "covered" else column
        assert serialized[column] is source[source_column]
    assert "result_schema_version" not in serialized
    assert "grid_strategy" not in serialized


def test_oracle_frame_projects_historical_expanded_rows() -> None:
    cleaned = clean_oracle_results_frame(pd.DataFrame([_expanded_row()]))
    assert list(cleaned.columns) == list(ORACLE_OUTPUT_COLUMNS)
    assert len(cleaned.columns) == 26
    assert cleaned.loc[0, "alpha_hat"] == 1.125
    assert bool(cleaned.loc[0, "covered"]) is True


def test_oracle_projection_rejects_missing_required_column() -> None:
    source = _expanded_row()
    del source["cr_length"]
    with pytest.raises(ValueError, match=r"missing required output columns.*cr_length"):
        serialize_oracle_result(source)


def test_current_oracle_csv_header_is_exact(tmp_path) -> None:
    output = tmp_path / "oracle.csv"
    clean_oracle_results_frame(pd.DataFrame([_expanded_row()])).to_csv(output, index=False)
    assert output.read_text(encoding="utf-8").splitlines()[0].split(",") == list(
        ORACLE_OUTPUT_COLUMNS
    )


def test_current_oracle_results_are_resumable_by_natural_key(tmp_path) -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",), n_values=(100,), p_values=(10,), pi_values=(0.5,),
        taus=(0.5,), reps=2, base_seed=12345,
    )
    existing = _expanded_row(rep=0)
    for column in ("dgp", "n", "p", "pi", "tau", "rep"):
        existing[column] = getattr(designs[0], column)
    path = tmp_path / "oracle.csv"
    clean_oracle_results_frame(pd.DataFrame([existing])).to_csv(path, index=False)
    assert filter_completed_designs(designs, path, ("oracle",)) == [designs[1]]
    assert filter_completed_designs(
        designs, path, ("oracle",), rerun_failed=True
    ) == [designs[1]]


def test_other_estimator_schema_constants_are_unchanged() -> None:
    assert "seed" in REQUIRED_DML_COLUMNS
    assert "result_schema_version" in REQUIRED_DML_COLUMNS
    assert "seed" in REQUIRED_POST_SELECTION_COLUMNS
    assert "result_schema_version" in REQUIRED_POST_SELECTION_COLUMNS
    assert tuple(REQUIRED_DML_COLUMNS) != ORACLE_OUTPUT_COLUMNS
    assert tuple(REQUIRED_POST_SELECTION_COLUMNS) != ORACLE_OUTPUT_COLUMNS


def test_current_oracle_file_loads_and_supports_thesis_metrics(tmp_path) -> None:
    rows = [_expanded_row(rep=0), _expanded_row(rep=1)]
    rows[1]["alpha_hat"] = 0.75
    rows[1]["cr_covers_true"] = False
    path = tmp_path / "oracle.csv"
    clean_oracle_results_frame(pd.DataFrame(rows)).to_csv(path, index=False)

    loaded = load_oracle_results(path, expected_replications=2)
    metrics = summarize_performance(loaded, ["estimator"]).iloc[0]
    assert metrics["bias"] == pytest.approx(-0.0625)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.125**2 + 0.25**2) / 2))
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["average_cr_length"] == pytest.approx(0.75)


@pytest.mark.slow
def test_small_deterministic_oracle_projection_preserves_scientific_results() -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, 0, 321)
    expanded = run_simulation_design(
        design, np.linspace(-1.0, 3.0, 5), estimators=("oracle",)
    )[0]
    compact = serialize_oracle_result(expanded)
    for column in ORACLE_OUTPUT_COLUMNS:
        source_column = "cr_covers_true" if column == "covered" else column
        if pd.isna(compact[column]) and pd.isna(expanded[source_column]):
            continue
        assert compact[column] == expanded[source_column]
