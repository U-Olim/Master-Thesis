from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dgp import Design, generate_data, true_active_control_indices
from estimators.dml import estimate_dml_ivqr
from estimators.oracle import estimate_oracle_ivqr
from estimators.post_selection import (
    estimate_post_selection_ivqr,
    select_controls_lasso,
)
from ivqr.confidence_regions import parse_cr_components
from simulation.dml_output import REQUIRED_DML_COLUMNS, clean_dml_results_frame
from simulation.oracle_output import ORACLE_OUTPUT_COLUMNS, clean_oracle_results_frame
from simulation.post_selection_output import (
    REQUIRED_POST_SELECTION_COLUMNS,
    clean_post_selection_results_frame,
)
from simulation.results import RESULT_COLUMNS, build_simulation_result_row
from simulation.runner import (
    make_design_seed,
    make_simulation_grid,
    run_simulation_design,
)
from utils.timing import RUNTIME_COLUMNS


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FLOAT_ATOL = 1e-12
REFERENCE_ALPHAS = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
REFERENCE_PARAMETERS = {
    "base_seed": 12345,
    "dgp": "dgp1",
    "n": 100,
    "p": 20,
    "pi": 0.5,
    "tau": 0.5,
    "rep": 0,
}
NONDETERMINISTIC_RESULT_COLUMNS = {
    "runtime_seconds",
    *RUNTIME_COLUMNS,
}


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _assert_expected_fields(actual: object, expected: dict[str, Any]) -> None:
    for field, expected_value in expected.items():
        actual_value = getattr(actual, field)
        if field == "cr_components":
            assert actual_value is not None
            np.testing.assert_allclose(
                np.asarray(actual_value),
                np.asarray(expected_value),
                rtol=0,
                atol=FLOAT_ATOL,
            )
        elif isinstance(expected_value, float):
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=0, atol=FLOAT_ATOL
            )
        else:
            assert actual_value == expected_value, field


def _reference_design() -> Design:
    seed = make_design_seed(**REFERENCE_PARAMETERS)
    return Design(
        REFERENCE_PARAMETERS["dgp"],
        REFERENCE_PARAMETERS["n"],
        REFERENCE_PARAMETERS["p"],
        REFERENCE_PARAMETERS["pi"],
        REFERENCE_PARAMETERS["tau"],
        REFERENCE_PARAMETERS["rep"],
        seed,
    )


def _direct_reference_results(design: Design) -> dict[str, object]:
    data = generate_data(design)
    estimator_random_state = design.seed % (2**32 - 1)
    common = {
        "grid_strategy": "adaptive",
        "refinement_tolerance": 0.025,
        "max_refinement_depth": 10,
        "max_alpha_evaluations": 201,
        "iteration_warning_policy": "use_if_valid",
        "hard_failure_policy": "unresolved",
        "adaptive_midpoint_probe": True,
        "alpha_hat_grid": "initial",
    }
    return {
        "oracle": estimate_oracle_ivqr(
            data,
            tau=design.tau,
            alphas=REFERENCE_ALPHAS,
            oracle_indices=true_active_control_indices(design.dgp, design.p),
            max_iter=1000,
            gmm_ridge=1e-8,
            critical_value_multiplier=1.0,
            **common,
        ),
        "post_selection_ivqr": estimate_post_selection_ivqr(
            data,
            tau=design.tau,
            alphas=REFERENCE_ALPHAS,
            selection_cv=3,
            selection_max_iter=10000,
            quantreg_max_iter=1000,
            selection_random_state=estimator_random_state,
            selection_lasso_multiplier=1.0,
            critical_value_multiplier=1.0,
            **common,
        ),
        "dml_ivqr": estimate_dml_ivqr(
            data,
            tau=design.tau,
            alphas=REFERENCE_ALPHAS,
            k_folds=3,
            fold_random_state=estimator_random_state,
            quantile_penalty=0.01,
            ridge_alpha=1.0,
            quantile_solver="highs",
            gmm_ridge=1e-8,
            critical_value_multiplier=1.0,
        ),
    }


@pytest.fixture(scope="module")
def reference_context() -> dict[str, Any]:
    design = _reference_design()
    direct_results = _direct_reference_results(design)
    runner_rows = run_simulation_design(
        design,
        REFERENCE_ALPHAS,
        estimators=("oracle", "post_selection", "dml"),
    )
    return {
        "design": design,
        "data": generate_data(design),
        "direct_results": direct_results,
        "runner_rows": runner_rows,
    }


def test_reference_design_seed_is_stable() -> None:
    fixture = _load_fixture("reference_seed.json")
    expected = fixture["expected_seed"]
    first = make_design_seed(**REFERENCE_PARAMETERS)
    second = make_design_seed(**REFERENCE_PARAMETERS)
    assert first == second == expected

    changed_rep = {**REFERENCE_PARAMETERS, "rep": 1}
    assert make_design_seed(**changed_rep) == fixture["rep_1_seed"]
    assert make_design_seed(**changed_rep) != expected

    full = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(100,),
        p_values=(20,),
        pi_values=(0.5,),
        taus=(0.5,),
        reps=4,
        base_seed=12345,
    )
    blocked = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(100,),
        p_values=(20,),
        pi_values=(0.5,),
        taus=(0.5,),
        reps=4,
        base_seed=12345,
        rep_start=2,
        rep_end=3,
    )
    assert [(item.rep, item.seed) for item in blocked] == [
        (item.rep, item.seed) for item in full[2:]
    ]


def test_reference_dgp_matches_frozen_values(reference_context: dict[str, Any]) -> None:
    fixture = _load_fixture("reference_dgp_values.json")
    data = reference_context["data"]
    arrays = {"y": data.y, "d": data.d, "z": data.z, "x": data.x}
    assert {name: list(value.shape) for name, value in arrays.items()} == fixture["shapes"]
    assert {name: str(value.dtype) for name, value in arrays.items()} == fixture["dtypes"]
    np.testing.assert_allclose(data.y[:5], fixture["y_first_5"], rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_array_equal(data.d[:10], fixture["d_first_10"])
    np.testing.assert_allclose(data.z[:5], fixture["z_first_5"], rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_allclose(
        data.x[:3, :4], fixture["x_first_3_first_4"], rtol=0, atol=FLOAT_ATOL
    )
    np.testing.assert_allclose(data.d.mean(), fixture["treatment_share"], rtol=0, atol=FLOAT_ATOL)
    for name, expected in fixture["means"].items():
        np.testing.assert_allclose(arrays[name].mean(), expected, rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_allclose(data.alpha_true, fixture["alpha_true"], rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_array_equal(
        true_active_control_indices("dgp1", 20), fixture["active_support"]
    )


def test_oracle_reference_result_matches_frozen_behavior(
    reference_context: dict[str, Any],
) -> None:
    fixture = _load_fixture("reference_oracle_result.json")
    result = reference_context["direct_results"]["oracle"]
    _assert_expected_fields(result, fixture["expected"])
    assert result.selected_controls == 5
    np.testing.assert_array_equal(
        true_active_control_indices("dgp1", 20), [0, 1, 2, 3, 4]
    )


def test_post_selection_reference_result_matches_frozen_behavior(
    reference_context: dict[str, Any],
) -> None:
    fixture = _load_fixture("reference_post_selection_result.json")
    result = reference_context["direct_results"]["post_selection_ivqr"]
    _assert_expected_fields(result, fixture["expected"])

    design = reference_context["design"]
    data = reference_context["data"]
    selected, message = select_controls_lasso(
        data.y,
        data.d,
        data.x,
        tau=design.tau,
        random_state=design.seed % (2**32 - 1),
        cv=3,
        max_iter=10000,
        selection_lasso_multiplier=1.0,
    )
    np.testing.assert_array_equal(selected, fixture["selected_support"])
    assert message == fixture["selection_message"]
    true_support = set(true_active_control_indices("dgp1", 20).tolist())
    selected_support = set(selected.tolist())
    assert len(true_support & selected_support) == 4
    assert len(selected_support - true_support) == 5
    assert len(true_support - selected_support) == 1
    assert not true_support.issubset(selected_support)


def test_dml_reference_result_matches_frozen_behavior(
    reference_context: dict[str, Any],
) -> None:
    fixture = _load_fixture("reference_dml_result.json")
    result = reference_context["direct_results"]["dml_ivqr"]
    _assert_expected_fields(result, fixture["expected"])
    assert fixture["configuration"]["k_folds"] == 3
    assert result.dml_qr_fit_count == len(REFERENCE_ALPHAS) * 3

    repeated = _direct_reference_results(reference_context["design"])["dml_ivqr"]
    _assert_expected_fields(repeated, fixture["expected"])


def test_runner_matches_direct_estimator(reference_context: dict[str, Any]) -> None:
    design = reference_context["design"]
    direct_rows = [
        build_simulation_result_row(design, result, REFERENCE_ALPHAS)
        for result in reference_context["direct_results"].values()
    ]
    direct = pd.DataFrame(direct_rows, columns=RESULT_COLUMNS)
    runner = pd.DataFrame(reference_context["runner_rows"], columns=RESULT_COLUMNS)
    retained = [
        column
        for column in RESULT_COLUMNS
        if column not in NONDETERMINISTIC_RESULT_COLUMNS
        and "runtime" not in column
    ]
    pd.testing.assert_frame_equal(
        direct.loc[:, retained],
        runner.loc[:, retained],
        check_dtype=False,
        check_exact=True,
    )


def test_current_serializers_preserve_real_internal_rows(
    reference_context: dict[str, Any],
) -> None:
    internal = pd.DataFrame(reference_context["runner_rows"], columns=RESULT_COLUMNS)
    cases = (
        (
            "oracle",
            clean_oracle_results_frame,
            ORACLE_OUTPUT_COLUMNS,
            {"covered": "cr_covers_true"},
        ),
        (
            "post_selection_ivqr",
            clean_post_selection_results_frame,
            REQUIRED_POST_SELECTION_COLUMNS,
            {
                "covered": "cr_covers_true",
                "n_selected_controls": "ps_n_selected_controls",
                "selection_lasso_multiplier": "ps_selection_lasso_multiplier",
                "selection_method": "ps_selection_method",
                "selection_target_y": "ps_selection_target_y",
                "selection_target_d": "ps_selection_target_d",
                "selection_quantile_specific": "ps_selection_quantile_specific",
                "instrument_selection_method": "ps_instrument_selection_method",
                "post_selection_inference_adjustment": "ps_post_selection_inference_adjustment",
                "n_retained_instruments": "ps_n_retained_instruments",
            },
        ),
        (
            "dml_ivqr",
            clean_dml_results_frame,
            REQUIRED_DML_COLUMNS,
            {"covered": "cr_covers_true"},
        ),
    )
    for estimator, cleaner, expected_columns, mappings in cases:
        source = internal.loc[internal["estimator"].eq(estimator)].reset_index(drop=True)
        cleaned = cleaner(source)
        assert tuple(cleaned.columns) == tuple(expected_columns)
        assert len(cleaned.columns) == {"oracle": 26, "post_selection_ivqr": 52, "dml_ivqr": 43}[estimator]
        for column in expected_columns:
            source_column = mappings.get(column, column)
            left = source.at[0, source_column]
            right = cleaned.at[0, column]
            if pd.isna(left) and pd.isna(right):
                continue
            assert left == right, (estimator, column)
        for removed in ("bias", "failed", "message", "runtime_seconds"):
            assert removed not in cleaned.columns
        if estimator == "oracle":
            assert "estimator" not in cleaned.columns
            assert "result_schema_version" not in cleaned.columns
        if pd.notna(cleaned.at[0, "cr_components"]):
            assert parse_cr_components(cleaned.at[0, "cr_components"]) is not None
