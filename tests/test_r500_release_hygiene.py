from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path

import pytest

from analysis.r500_release_hygiene import (
    ATTRIBUTE_TARGETS,
    PHASE5_FILENAMES,
    RAW_CONTRACTS,
    effective_attributes,
    experimental_documents_review,
    finalize_release_hygiene,
    git_attribute_checks,
    gitignore_checks,
    raw_integrity,
    release_metadata_status,
    stable_json,
    windows_checkout_preservation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / "results/thesis/r500/review"
MODULE = PROJECT_ROOT / "src/analysis/r500_release_hygiene.py"
SCRIPT = PROJECT_ROOT / "scripts/finalize_r500_release_hygiene.py"


@pytest.fixture(scope="module")
def generated_status() -> dict[str, object]:
    return finalize_release_hygiene(PROJECT_ROOT)


def test_gitattributes_exists_and_is_minimal() -> None:
    text = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto" in text
    assert "working-tree-encoding" not in text


@pytest.mark.parametrize("relative", tuple(RAW_CONTRACTS))
def test_protected_raw_files_disable_text_and_eol(relative: str) -> None:
    attrs = effective_attributes(PROJECT_ROOT, relative)
    assert attrs["text"] == "unset"
    assert attrs["eol"] == "unset"


@pytest.mark.parametrize("suffix", ("pdf", "png"))
def test_rendered_artifacts_are_binary(suffix: str) -> None:
    relative = next(path for path in ATTRIBUTE_TARGETS if path.endswith(f".{suffix}"))
    attrs = effective_attributes(PROJECT_ROOT, relative)
    assert attrs["binary"] == "set"
    assert attrs["text"] == "unset"


@pytest.mark.parametrize("relative", ("README.md", "scripts/build_r500_thesis_package.py"))
def test_source_files_use_lf(relative: str) -> None:
    attrs = effective_attributes(PROJECT_ROOT, relative)
    assert attrs["text"] == "set"
    assert attrs["eol"] == "lf"


@pytest.mark.parametrize(
    ("relative", "eol"),
    (
        ("results/validation/r500_audit/structural_validation.json", "crlf"),
        ("results/validation/r500_phase2/coverage_uncertainty.csv", "lf"),
        ("results/thesis/r500/thesis_findings.json", "crlf"),
    ),
)
def test_generated_phase_files_remain_text(relative: str, eol: str) -> None:
    attrs = effective_attributes(PROJECT_ROOT, relative)
    assert attrs["text"] == "set"
    assert attrs["eol"] == eol
    assert attrs.get("binary") != "set"


def test_no_broad_csv_or_json_binary_rule() -> None:
    rules = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("*.csv ") and "-text" in line for line in rules)
    assert not any(line.startswith("*.json ") and "-text" in line for line in rules)
    assert all(row["status"] == "pass" for row in git_attribute_checks(PROJECT_ROOT))


def test_expected_raw_hashes_sizes_rows_and_columns() -> None:
    assert all(row["status"] == "pass" for row in raw_integrity(PROJECT_ROOT))


def test_windows_checkout_preserves_protected_hashes_without_restoration() -> None:
    result = windows_checkout_preservation(PROJECT_ROOT)
    assert result["status"] == "pass"
    assert result["all_hashes_preserved"] is True
    assert result["manual_restoration_required"] is False


def test_gitignore_required_cache_patterns_and_no_broad_hides() -> None:
    checks = gitignore_checks(PROJECT_ROOT)
    assert all(row["status"] == "pass" for row in checks)


def test_gitignore_keeps_manifest_and_thesis_outputs_visible() -> None:
    checks = {row["check"]: row for row in gitignore_checks(PROJECT_ROOT)}
    assert checks["protected_manifest_visible"]["status"] == "pass"
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/\n" not in text
    assert "documents/\n" not in text


def test_phase4_safe_cache_scope_has_no_tracked_deletion() -> None:
    rows = list(csv.DictReader((REVIEW_DIR / "repository_hygiene.csv").open(encoding="utf-8")))
    removable = [row for row in rows if row["safe_to_remove"] == "True"]
    assert removable
    assert all(row["tracked"] == "False" for row in removable)
    assert all(row["recommended_action"] == "remove_after_review" for row in removable)


def test_experimental_qmd_and_pdf_classifications() -> None:
    rows = {row["path"]: row for row in experimental_documents_review(PROJECT_ROOT)}
    assert rows["documents/Experiments.qmd"]["recommended_action"] == "keep_with_warning"
    assert rows["documents/Experiments.pdf"]["recommended_action"] == "keep_with_warning"
    assert rows["documents/Experiments.pdf"]["matches_repository_state"] == "unverified_pdf_predates_qmd"


def test_license_and_citation_absence_require_owner_decisions() -> None:
    status = release_metadata_status(PROJECT_ROOT)
    assert status["license_present"] is False
    assert status["citation_present"] is False
    assert status["license_decision_required"] is True
    assert status["citation_decision_required"] is True
    assert status["blocking_for_private_submission"] is False


def test_release_metadata_contains_no_fabricated_identity() -> None:
    status = release_metadata_status(PROJECT_ROOT)
    payload = stable_json(status)
    assert "ORCID if applicable" in payload
    assert not any(key in status for key in ("authors", "doi", "supervisor", "release_date"))


def test_release_metadata_json_is_deterministic() -> None:
    status = release_metadata_status(PROJECT_ROOT)
    assert stable_json(status) == stable_json(status)
    assert hashlib.sha256(stable_json(status).encode()).digest() == hashlib.sha256(stable_json(status).encode()).digest()


def test_phase5_generation_is_deterministic(generated_status: dict[str, object]) -> None:
    before = {
        filename: ((REVIEW_DIR / filename).stat().st_size, hashlib.sha256((REVIEW_DIR / filename).read_bytes()).hexdigest())
        for filename in PHASE5_FILENAMES
    }
    assert finalize_release_hygiene(PROJECT_ROOT) == generated_status
    after = {
        filename: ((REVIEW_DIR / filename).stat().st_size, hashlib.sha256((REVIEW_DIR / filename).read_bytes()).hexdigest())
        for filename in PHASE5_FILENAMES
    }
    assert before == after


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_import_contract_excludes_scientific_modules() -> None:
    prohibited = ("dgp", "estimators", "ivqr", "simulation")
    for path in (MODULE, SCRIPT):
        assert not any(
            name == prefix or name.startswith(f"{prefix}.")
            for name in _imports(path)
            for prefix in prohibited
        )


def test_validator_has_no_simulation_or_tag_side_effect_commands() -> None:
    source = MODULE.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert "git tag" not in source
    assert "run_oracle_ivqr" not in source
    assert "run_post_selection_ivqr" not in source
    assert "run_dml_ivqr" not in source


def test_phase1_phase2_phase3_phase4_integrity(generated_status: dict[str, object]) -> None:
    for key in ("phase1_integrity", "phase2_integrity", "phase3_integrity", "phase4_integrity"):
        assert generated_status[key] == "pass"
    rows = list(csv.DictReader((REVIEW_DIR / "final_integrity_checks.csv").open(encoding="utf-8")))
    assert len([row for row in rows if row["phase"] == "phase3" and row["check"] != "all_190_consistency_checks"]) == 35
    assert all(row["status"] == "pass" for row in rows)


def test_final_release_classification(generated_status: dict[str, object]) -> None:
    assert generated_status["classification"] == "ready_with_owner_decisions"
    assert generated_status["manual_byte_restoration_required"] is False
    assert generated_status["scientific_code_changed"] is False
    assert generated_status["simulation_ran"] is False
    assert generated_status["tag_created"] is False


def test_outputs_have_no_volatile_timestamps_or_absolute_paths(generated_status: dict[str, object]) -> None:
    del generated_status
    for filename in PHASE5_FILENAMES:
        text = (REVIEW_DIR / filename).read_text(encoding="utf-8")
        assert "C:\\Users\\" not in text
        assert "generation_timestamp" not in text
        assert "current_time" not in text


def test_phase5_output_contract_is_exact(generated_status: dict[str, object]) -> None:
    del generated_status
    assert len(PHASE5_FILENAMES) == 8
    assert all((REVIEW_DIR / filename).is_file() for filename in PHASE5_FILENAMES)
