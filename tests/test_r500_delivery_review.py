from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.r500_delivery_review import (
    REVIEW_FILENAMES,
    archive_recommendations,
    build_caption_registry,
    classify_delivery,
    compare_integrity_records,
    compile_latex_table,
    expected_phase3_files,
    parse_simple_latex_table,
    protected_integrity,
    repository_hygiene,
    review_figures,
    review_phase3_inventory,
    review_readme,
    review_report,
    review_tables,
    stable_json,
    terminology_checks,
    validate_consistency,
    validate_static_latex,
)
from analysis.r500_thesis_package import (
    ESTIMATOR_ORDER,
    FIGURE_FAMILIES,
    PAIR_ORIENTATION,
    TABLE_FAMILIES,
    dataframe_to_latex,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "results" / "thesis" / "r500"
MODULE = PROJECT_ROOT / "src" / "analysis" / "r500_delivery_review.py"
SCRIPT = PROJECT_ROOT / "scripts" / "review_r500_thesis_delivery.py"


def _synthetic_package(tmp_path: Path) -> Path:
    package = tmp_path / "r500"
    package.mkdir()
    outputs = []
    for index in range(34):
        relative = f"artifact_{index:02d}.txt"
        path = package / relative
        path.write_text(str(index), encoding="utf-8")
        outputs.append(
            {
                "output_filename": relative,
                "generated_output_sha256": sha256_file(path),
            }
        )
    outputs.append(
        {
            "output_filename": "thesis_output_manifest.json",
            "generated_output_sha256": None,
        }
    )
    (package / "thesis_output_manifest.json").write_text(
        json.dumps({"outputs": outputs}), encoding="utf-8"
    )
    return package


def test_expected_inventory_and_manifest_hash_validation(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    manifest = json.loads((package / "thesis_output_manifest.json").read_text())
    assert len(expected_phase3_files(manifest)) == 35
    inventory, unexpected = review_phase3_inventory(package)
    assert unexpected == []
    assert inventory.status.eq("pass").all()


def test_missing_and_unexpected_phase3_files_are_reported(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    (package / "artifact_00.txt").unlink()
    (package / "unexpected.tmp").write_text("x", encoding="utf-8")
    inventory, unexpected = review_phase3_inventory(package)
    assert unexpected == ["unexpected.tmp"]
    assert inventory.status.eq("fail").any()


def test_manifest_hash_mismatch_fails_inventory(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    (package / "artifact_01.txt").write_text("changed", encoding="utf-8")
    inventory, _ = review_phase3_inventory(package)
    row = inventory[inventory.filename.eq("artifact_01.txt")].iloc[0]
    assert row.hash_status == "fail"
    assert row.status == "fail"


def test_csv_latex_pairs_rows_values_ordering_and_dml_availability() -> None:
    reviews = review_tables(PACKAGE_DIR)
    assert reviews.table_id.tolist() == list(TABLE_FAMILIES)
    assert reviews.status.eq("pass").all()
    overall = pd.read_csv(PACKAGE_DIR / "tables/table_02_overall_estimator_performance.csv")
    assert overall.estimator.tolist() == list(ESTIMATOR_ORDER)
    dml = overall[overall.estimator.eq("dml")].iloc[0]
    assert pd.isna(dml.iteration_warning_rate)
    assert pd.isna(dml.unresolved_rate)


def test_simple_latex_parser_and_escaping(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a_b": ["Post-selection", "x&y"], "value": [0.25, None]})
    path = tmp_path / "table.tex"
    path.write_text(dataframe_to_latex(frame), encoding="utf-8")
    columns, rows = parse_simple_latex_table(path)
    assert columns == ["a_b", "value"]
    assert rows == [["Post-selection", "0.2500"], ["x&y", "N/A"]]
    assert validate_static_latex(path.read_text())["status"] == "pass"


@pytest.mark.parametrize(
    "text,field,expected",
    [
        (r"\begin{tabular}{l} x \\ \end{tabular}", "status", "pass"),
        (r"\begin{tabular}{l x \\ \end{tabular}", "balanced_braces", False),
        (r"\begin{tabular}{l} x \\ \end{table}", "balanced_environments", False),
        (r"\begin{tabular}{l} 5% \\ \end{tabular}", "escaped_special_characters", False),
    ],
)
def test_static_latex_validation_detects_structure_and_escaping(
    text: str, field: str, expected: object
) -> None:
    result = validate_static_latex(text)
    assert result[field] == expected


def test_latex_compiler_unavailable_and_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "table.tex"
    table.write_text(dataframe_to_latex(pd.DataFrame({"x": [1]})), encoding="utf-8")
    monkeypatch.setattr("analysis.r500_delivery_review.shutil.which", lambda _: None)
    assert compile_latex_table(table)["status"] == "not_available"
    assert compile_latex_table(table, engine="definitely_missing_engine")["status"] == "fail"


def test_generated_pdf_png_readability_dimensions_and_manifest_hashes() -> None:
    manifest = json.loads((PACKAGE_DIR / "thesis_output_manifest.json").read_text())
    reviews = review_figures(PACKAGE_DIR, manifest)
    assert reviews.figure_id.tolist() == list(FIGURE_FAMILIES)
    assert reviews.pdf_readable.all()
    assert reviews.png_readable.all()
    assert reviews.manifest_hashes.eq("pass").all()
    assert reviews.page_count.ge(1).all()


def test_paired_orientation_is_consistent_in_table_figure_and_report() -> None:
    paired = pd.read_csv(PACKAGE_DIR / "tables/table_03_estimator_tradeoffs.csv")
    assert set(paired.difference_orientation) == {PAIR_ORIENTATION}
    report = (PACKAGE_DIR / "empirical_results_report.md").read_text()
    assert "relative to" in report
    captions = build_caption_registry()["entries"]
    paired_entries = [item for item in captions if item["id"] in {
        "table_03_estimator_tradeoffs", "figure_06_paired_estimator_differences"
    }]
    assert all(PAIR_ORIENTATION in item["interpretation_note"] for item in paired_entries)


def test_terminology_causal_language_and_multiplier_disclosures(tmp_path: Path) -> None:
    checks = terminology_checks(PACKAGE_DIR, PROJECT_ROOT / "README.md")
    assert not checks.status.eq("fail").any()
    report = (PACKAGE_DIR / "empirical_results_report.md").read_text()
    assert "multiplier 1.0" in report
    assert "no conclusion about 1.8" in report
    bad_readme = tmp_path / "README.md"
    bad_readme.write_text("Post Selection caused zero warnings.", encoding="utf-8")
    bad = terminology_checks(PACKAGE_DIR, bad_readme)
    assert bad.status.eq("fail").any()


def test_findings_and_numerical_report_provenance_integration() -> None:
    assert review_report(PACKAGE_DIR)["status"] == "pass"
    result = validate_consistency(PACKAGE_DIR)
    assert result == {
        "check_count": 190,
        "all_checks_pass": True,
        "all_findings_sourced": True,
        "status": "pass",
    }


def test_integrity_comparison_detects_raw_phase1_and_phase2_mutation() -> None:
    before = [{"path": "x", "sha256": "a", "bytes": 1, "modified_ns": 1}]
    assert compare_integrity_records(before, before) == []
    after = [{"path": "x", "sha256": "b", "bytes": 1, "modified_ns": 1}]
    assert compare_integrity_records(before, after) == ["x"]
    current = protected_integrity(PROJECT_ROOT)
    assert current["status"] == "pass"
    authorities = {item["authority"] for item in current["records"]}
    assert "results/raw/manifest.json" in authorities
    assert "git HEAD / Phase 1" in authorities
    assert "git HEAD / Phase 2" in authorities


def test_repository_hygiene_and_archive_recommendations_are_non_destructive() -> None:
    hygiene = repository_hygiene(PROJECT_ROOT)
    assert set(hygiene.recommended_action).issubset(
        {"keep", "ignore", "archive", "remove_after_review", "investigate"}
    )
    archive = archive_recommendations(PROJECT_ROOT)
    assert not archive.recommended_action.isin({"remove", "delete"}).any()
    stub = archive[archive.path.eq("scenarios/run_simulation.py")].iloc[0]
    assert stub.recommended_action == "keep"


def _classification_inputs(scientific: str = "pass", latex: str = "pass") -> dict[str, object]:
    return {
        "inventory": pd.DataFrame({"status": ["pass"]}),
        "tables": pd.DataFrame({"status": ["pass"]}),
        "figures": pd.DataFrame({"status": ["pass"]}),
        "latex": pd.DataFrame({"status": [latex]}),
        "consistency": {"status": scientific},
        "integrity": {"status": "pass"},
        "terminology": pd.DataFrame({"status": ["pass"]}),
        "report": {"status": "pass"},
    }


def test_delivery_classification_and_scientific_failure() -> None:
    inputs = _classification_inputs()
    assert classify_delivery(**inputs)["overall"] == "ready_with_minor_corrections"
    unavailable = _classification_inputs(latex="not_available")
    result = classify_delivery(**unavailable)
    assert result["latex_readiness"] == "not_available"
    assert result["overall"] == "ready_with_minor_corrections"
    failed = _classification_inputs(scientific="fail")
    assert classify_delivery(**failed)["overall"] == "not_ready"


def test_readme_reproduction_contract_is_complete() -> None:
    result = review_readme(PROJECT_ROOT / "README.md")
    assert result["status"] == "pass"
    assert result["passed"] == result["total"] == 20


def test_caption_registry_is_complete_and_noncausal() -> None:
    registry = build_caption_registry()
    assert registry["entry_count"] == 15
    assert {item["id"] for item in registry["entries"]} == set(TABLE_FAMILIES) | set(FIGURE_FAMILIES)
    text = stable_json(registry)
    assert not any(word in text.lower() for word in (" caused ", " proved ", " guaranteed "))


def test_phase4_output_serialization_is_deterministic() -> None:
    payload = {"b": 2, "a": [1, None]}
    assert stable_json(payload) == stable_json(payload)
    assert hashlib.sha256(stable_json(payload).encode()).hexdigest() == hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    assert len(REVIEW_FILENAMES) == 12


def test_phase4_import_contract_excludes_scientific_modules() -> None:
    prohibited = ("dgp", "estimators", "ivqr", "simulation")
    for path in (MODULE, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == prefix or name.startswith(f"{prefix}.") for name in imports for prefix in prohibited)
