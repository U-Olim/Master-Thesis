from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.r500_thesis_package import (
    CSV_MISSING,
    ESTIMATOR_ORDER,
    FIGURE_FAMILIES,
    PAIR_ORIENTATION,
    SOURCE_CONTRACTS,
    TABLE_FAMILIES,
    build_design_table,
    build_overall_table,
    build_tradeoff_table,
    build_warning_exception_table,
    consistency_row,
    dataframe_to_latex,
    finding,
    format_value,
    require_consistency,
    scenario_sort,
    sha256_file,
    sourced_sentence,
    validate_findings,
    validate_historical_multiplier_outputs,
    validate_report_provenance,
    validate_source_contracts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "results" / "thesis" / "r500"


def _load_build_script():
    path = PROJECT_ROOT / "scripts" / "build_r500_thesis_package.py"
    spec = importlib.util.spec_from_file_location("build_r500_thesis_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structural() -> dict[str, object]:
    common = {
        "rows": 8,
        "unique_replications": 2,
        "design_cells": 4,
    }
    return {
        "structure": {name: common.copy() for name in ESTIMATOR_ORDER},
        "artifacts": {"oracle": {"columns": ["dgp", "n"]}},
    }


def _scenario_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp2", "dgp1"],
            "n": [1000, 500],
            "p": [500, 200],
            "pi": [1.0, 0.1],
            "tau": [0.75, 0.25],
        }
    )


def test_authoritative_source_contracts_and_hashes_validate() -> None:
    hashes = validate_source_contracts(PROJECT_ROOT)
    assert hashes == {
        path: contract["sha256"] for path, contract in SOURCE_CONTRACTS.items()
    }


def test_design_table_derives_values_and_totals() -> None:
    row = build_design_table(_structural(), _scenario_values()).iloc[0]
    assert row["dgp_values"] == "dgp1; dgp2"
    assert row["sample_sizes"] == "500; 1000"
    assert row["replications_per_design_cell"] == 2
    assert row["design_cells"] == 4
    assert row["rows_per_estimator"] == 8
    assert row["total_estimator_rows"] == 24


def test_overall_table_preserves_values_order_and_unavailable_dml() -> None:
    summary = pd.DataFrame(
        {
            "estimator": ["dml", "oracle", "post_selection"],
            "conditional_coverage": [0.95, 0.94, 0.92],
            "coverage_denominator": [9, 10, 8],
            "bias": [0.0, 0.1, -0.1],
            "rmse": [1.2, 1.0, 1.1],
            "mean_cr_length": [3.0, 2.0, 2.1],
            "empty_valid_rate": [np.nan, 0.0, 0.01],
            "unresolved_rate": [np.nan, 0.0, 0.0],
            "iteration_warning_rate": [np.nan, 0.7, 0.9],
        }
    )
    classification = pd.DataFrame(
        {
            "estimator": ["oracle", "post_selection", "dml"],
            "overall_wilson95_lower": [0.9, 0.8, 0.9],
            "overall_wilson95_upper": [0.97, 0.95, 0.98],
            "coverage_gap": [-0.01, -0.03, 0.0],
            "classification": ["acceptable", "concerning", "acceptable"],
            "diagnostic_confidence": ["high", "high", "limited"],
        }
    )
    result = build_overall_table(summary, classification)
    assert result["estimator"].tolist() == list(ESTIMATOR_ORDER)
    assert result.loc[0, "conditional_coverage"] == pytest.approx(0.94)
    assert np.isnan(result.loc[2, "iteration_warning_rate"])


def test_tradeoff_table_retains_phase2_orientation_and_denominators() -> None:
    rows = []
    for metric, value in (("coverage", 0.1), ("squared_error", -0.2), ("cr_length", 0.3)):
        rows.append({
            "estimator_a": "oracle", "estimator_b": "post_selection", "metric": metric,
            "mean_paired_difference": value, "paired_ci95_lower": value - 0.01,
            "paired_ci95_upper": value + 0.01, "valid_paired_denominator": 99,
        })
    result = build_tradeoff_table(pd.DataFrame(rows)).iloc[0]
    assert result["difference_orientation"] == PAIR_ORIENTATION
    assert result["paired_coverage_difference"] == pytest.approx(0.1)
    assert result["paired_squared_error_difference"] == pytest.approx(-0.2)
    assert result["paired_cr_length_denominator"] == 99


def test_warning_exception_table_never_substitutes_dml_zero() -> None:
    warning = pd.DataFrame(
        {
            "estimator": ["oracle", "post_selection"],
            "warning_category": ["iteration_warning"] * 2,
            "warning_frequency": [0.7, 0.9],
            "warning_event_count": [10, 20],
            "coverage_affected_valid": [0.9, 0.8],
            "coverage_without_warning_valid": [0.95, 0.94],
            "rmse": [1.0, 1.1],
            "rmse_without_warning": [0.9, 1.0],
        }
    )
    exceptions = pd.DataFrame(
        {
            "estimator_name": ["oracle", "post_selection", "dml", "dml"],
            "exception_type": ["empty_cr", "unresolved_cr", "missing_legacy_geometry", "missing_legacy_geometry"],
        }
    )
    result = build_warning_exception_table(warning, exceptions).set_index("estimator")
    assert np.isnan(result.loc["dml", "warning_row_frequency"])
    assert np.isnan(result.loc["dml", "unresolved_rows"])
    assert result.loc["dml", "missing_legacy_geometry"] == 2


def test_deterministic_scenario_sort_uses_estimator_and_natural_key_order() -> None:
    frame = pd.DataFrame(
        {
            "estimator": ["dml", "oracle", "oracle"],
            "dgp": ["dgp1", "dgp2", "dgp1"], "n": [500] * 3,
            "p": [200] * 3, "pi": [0.1] * 3, "tau": [0.5] * 3,
        }
    )
    result = scenario_sort(frame)
    assert result[["estimator", "dgp"]].values.tolist() == [
        ["oracle", "dgp1"], ["oracle", "dgp2"], ["dml", "dgp1"]
    ]


def test_explicit_formatting_and_latex_escaping() -> None:
    assert format_value(np.nan) == CSV_MISSING
    assert format_value(np.nan, latex=True) == "N/A"
    assert format_value(0.5) == "0.5000"
    latex = dataframe_to_latex(pd.DataFrame({"a_b": ["x&y", np.nan]}))
    assert "a\\_b" in latex
    assert "x\\&y" in latex
    assert "N/A" in latex


def test_findings_require_valid_provenance_and_reject_causal_wording() -> None:
    source = next(iter(SOURCE_CONTRACTS))
    valid = finding(
        "f1", "topic", {"value": 1}, source, SOURCE_CONTRACTS[source]["sha256"],
        {}, 1, "descriptive", "Value was higher.", "Do not infer causation."
    )
    validate_findings([valid])
    invalid = dict(valid, id="f2", permitted_wording="This caused the outcome.")
    with pytest.raises(ValueError, match="prohibited causal wording"):
        validate_findings([invalid])


def test_report_numerical_sentence_provenance_is_structural() -> None:
    report = sourced_sentence("Coverage was 0.9500.", ["coverage"])
    validate_report_provenance(report, {"coverage"})
    with pytest.raises(ValueError, match="lacks finding provenance"):
        validate_report_provenance("Coverage was 0.9500.", {"coverage"})


def test_consistency_check_fails_on_substantive_difference() -> None:
    passed = consistency_row("x", "coverage", "source.csv", 0.95, 0.9500001, 1e-3)
    require_consistency(pd.DataFrame([passed]))
    failed = consistency_row("x", "coverage", "source.csv", 0.95, 0.90, 1e-3)
    with pytest.raises(ValueError, match="consistency checks failed"):
        require_consistency(pd.DataFrame([failed]))


def test_deterministic_text_output_bytes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"value": [0.5, np.nan], "label": ["a", "b"]})
    first = tmp_path / "first.tex"
    second = tmp_path / "second.tex"
    first.write_text(dataframe_to_latex(frame), encoding="utf-8", newline="\n")
    second.write_text(dataframe_to_latex(frame), encoding="utf-8", newline="\n")
    assert first.read_bytes() == second.read_bytes()


def test_pdf_and_png_metadata_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_build_script()
    monkeypatch.setattr(script, "FIGURE_DIR", tmp_path)
    hashes = []
    for _ in range(2):
        figure, axis = plt.subplots(figsize=(3, 2))
        axis.plot([0, 1], [0, 1])
        paths = script._save_figure(figure, "deterministic")
        hashes.append(tuple(sha256_file(path) for path in paths))
    assert hashes[0] == hashes[1]
    assert b"CreationDate" not in (tmp_path / "deterministic.pdf").read_bytes()
    assert b"date:create" not in (tmp_path / "deterministic.png").read_bytes()


def test_historical_multiplier_validator_accepts_disclosure_and_rejects_misattribution(tmp_path: Path) -> None:
    good = tmp_path / "good.md"
    good.write_text(
        "Historical Post-selection results use multiplier 1.0; no conclusion about 1.8 follows.",
        encoding="utf-8",
    )
    validate_historical_multiplier_outputs(tmp_path)
    bad = tmp_path / "bad.md"
    bad.write_text("Historical results use multiplier 1.8.", encoding="utf-8")
    with pytest.raises(ValueError, match="misattribution"):
        validate_historical_multiplier_outputs(tmp_path)


def test_phase3_import_contract_excludes_scientific_packages() -> None:
    paths = [
        PROJECT_ROOT / "src" / "analysis" / "r500_thesis_package.py",
        PROJECT_ROOT / "scripts" / "build_r500_thesis_package.py",
    ]
    prohibited = {"estimators", "dgp", "ivqr", "simulation.execution", "src.estimators", "src.dgp", "src.ivqr"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert imports.isdisjoint(prohibited)


def test_generated_package_has_exact_families_and_complete_manifest() -> None:
    manifest = json.loads((PACKAGE_DIR / "thesis_output_manifest.json").read_text())
    assert manifest["table_family_count"] == 7
    assert manifest["figure_family_count"] == 8
    table_files = sorted((PACKAGE_DIR / "tables").glob("*"))
    figure_files = sorted((PACKAGE_DIR / "figures").glob("*"))
    assert len(table_files) == 14
    assert len(figure_files) == 16
    assert {path.stem for path in table_files} == set(TABLE_FAMILIES)
    assert {path.stem for path in figure_files} == set(FIGURE_FAMILIES)
    actual = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file()
    }
    recorded = {entry["output_filename"] for entry in manifest["outputs"]}
    assert actual == recorded
    for entry in manifest["outputs"]:
        if entry["output_filename"] != "thesis_output_manifest.json":
            assert entry["generated_output_sha256"]
        assert entry["authoritative_sources"]


def test_generated_disclosures_and_unavailable_values_are_explicit() -> None:
    report = (PACKAGE_DIR / "empirical_results_report.md").read_text(encoding="utf-8")
    manifest = json.loads((PACKAGE_DIR / "thesis_output_manifest.json").read_text())
    assert "multiplier 1.0" in report
    assert "no conclusion about 1.8" in report
    assert "15 columns" in report
    assert "not zeros" in report
    assert manifest["historical_post_selection_multiplier"] == 1.0
    assert manifest["future_multiplier_not_analyzed"] == 1.8
    table_csv = (PACKAGE_DIR / "tables" / "table_07_warning_exception_diagnostics.csv").read_text()
    table_tex = (PACKAGE_DIR / "tables" / "table_07_warning_exception_diagnostics.tex").read_text()
    dml_line = next(line for line in table_csv.splitlines() if line.startswith("dml,"))
    assert ",NA,NA,NA,NA,NA,NA,NA,NA,43," in dml_line
    assert "N/A" in table_tex
    validate_historical_multiplier_outputs(PACKAGE_DIR)


def test_findings_and_report_have_complete_source_provenance() -> None:
    payload = json.loads((PACKAGE_DIR / "thesis_findings.json").read_text())
    validate_findings(payload["findings"])
    report = (PACKAGE_DIR / "empirical_results_report.md").read_text(encoding="utf-8")
    validate_report_provenance(report, {item["id"] for item in payload["findings"]})


def test_consistency_outputs_all_pass() -> None:
    checks = pd.read_csv(PACKAGE_DIR / "consistency_checks.csv")
    assert len(checks) >= 1
    assert checks["status"].eq("passed").all()
    assert "All" in (PACKAGE_DIR / "consistency_report.md").read_text()


def test_source_validation_does_not_mutate_raw_phase1_or_phase2() -> None:
    paths = [
        *sorted((PROJECT_ROOT / "results" / "raw").glob("*")),
        *sorted((PROJECT_ROOT / "results" / "validation" / "r500_audit").rglob("*")),
        *sorted((PROJECT_ROOT / "results" / "validation" / "r500_phase2").rglob("*")),
    ]
    files = [path for path in paths if path.is_file()]
    before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in files}
    validate_source_contracts(PROJECT_ROOT)
    after = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in files}
    assert after == before
