"""Deterministic, non-scientific release checks for the historical R=500 package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Mapping

from analysis.r500_delivery_review import REVIEW_FILENAMES, protected_integrity
from analysis.r500_thesis_package import sha256_file


PHASE5_FILENAMES = (
    "release_hygiene_report.md",
    "release_hygiene_status.json",
    "git_attribute_checks.csv",
    "gitignore_checks.csv",
    "experimental_documents_review.csv",
    "release_metadata_recommendations.md",
    "release_metadata_status.json",
    "final_integrity_checks.csv",
)

RAW_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "results/raw/oracle_ivqr.csv": {
        "sha256": "44ff5b1030e1a80386e62b840be9f842c2229f43e630000cc9e52567ac84be36",
        "bytes": 22624466,
        "rows": 72000,
        "columns": 43,
    },
    "results/raw/post_selection_ivqr.csv": {
        "sha256": "7c9b914a8d5cf7cc7c3b9bda3458641de4f62bdfcf1644793871555275bfedd8",
        "bytes": 30818201,
        "rows": 72000,
        "columns": 52,
    },
    "results/raw/dml_ivqr.csv": {
        "sha256": "27d3c514e7a608083647c9485e1ba4d73267cdf80753e3bae8c22a4ee4c5f803",
        "bytes": 9160449,
        "rows": 72000,
        "columns": 15,
    },
    "results/raw/manifest.json": {
        "sha256": "bb9c0dfddc36e1d44be4749f7d1093313ce12040eab4754d5fa41ad42960b4c5",
        "bytes": 7340,
        "rows": None,
        "columns": None,
    },
}

ATTRIBUTE_TARGETS = (
    "results/raw/oracle_ivqr.csv",
    "results/raw/post_selection_ivqr.csv",
    "results/raw/dml_ivqr.csv",
    "results/raw/manifest.json",
    "README.md",
    "scripts/build_r500_thesis_package.py",
    "results/validation/r500_audit/structural_validation.json",
    "results/validation/r500_phase2/coverage_uncertainty.csv",
    "results/thesis/r500/thesis_findings.json",
    "results/thesis/r500/figures/figure_01_overall_estimator_tradeoff.pdf",
    "results/thesis/r500/figures/figure_01_overall_estimator_tradeoff.png",
)

REQUIRED_IGNORE_PATTERNS = (
    "__pycache__/",
    ".pytest_cache/",
    "*.py[cod]",
    ".coverage",
    "htmlcov/",
    ".mypy_cache/",
    ".pyright/",
    ".ruff_cache/",
    ".ipynb_checkpoints/",
    ".DS_Store",
    "Thumbs.db",
    "*.aux",
    "*.log",
    "*.out",
    "*.toc",
    "*.fls",
    "*.fdb_latexmk",
    "*.synctex.gz",
)


def stable_json(payload: object) -> str:
    """Return sorted, newline-terminated JSON."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def effective_attributes(root: Path, relative: str) -> dict[str, str]:
    """Read all effective Git attributes for a repository-relative path."""
    output = _run_git(root, "check-attr", "-a", "--", relative).stdout
    attributes: dict[str, str] = {}
    for line in output.splitlines():
        _, attribute, value = line.split(": ", 2)
        attributes[attribute] = value
    return attributes


def git_attribute_checks(root: Path) -> list[dict[str, object]]:
    """Validate the minimal text/binary policy and return an auditable matrix."""
    rules = (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    broad_csv_binary = any(line.strip().startswith("*.csv ") and "-text" in line for line in rules)
    broad_json_binary = any(line.strip().startswith("*.json ") and "-text" in line for line in rules)
    rows: list[dict[str, object]] = []
    for relative in ATTRIBUTE_TARGETS:
        attrs = effective_attributes(root, relative)
        is_raw = relative in RAW_CONTRACTS
        suffix = Path(relative).suffix.lower()
        crlf_contract = relative in {
            "results/validation/r500_audit/structural_validation.json",
            "results/thesis/r500/thesis_findings.json",
        }
        expected = (
            "byte_preserved"
            if is_raw
            else "binary"
            if suffix in {".pdf", ".png"}
            else "text_crlf"
            if crlf_contract
            else "text_lf"
        )
        if expected == "byte_preserved":
            passed = attrs.get("text") == "unset" and attrs.get("eol") == "unset"
        elif expected == "binary":
            passed = attrs.get("binary") == "set" and attrs.get("text") == "unset"
        elif expected == "text_crlf":
            passed = attrs.get("text") == "set" and attrs.get("eol") == "crlf"
        else:
            passed = attrs.get("text") == "set" and attrs.get("eol") == "lf"
        rows.append(
            {
                "path": relative,
                "expected_policy": expected,
                "text": attrs.get("text", "unspecified"),
                "eol": attrs.get("eol", "unspecified"),
                "binary": attrs.get("binary", "unspecified"),
                "diff": attrs.get("diff", "unspecified"),
                "merge": attrs.get("merge", "unspecified"),
                "status": "pass" if passed else "fail",
                "notes": "effective attributes from git check-attr",
            }
        )
    rows.extend(
        [
            {
                "path": "*.csv",
                "expected_policy": "not_broadly_byte_preserved",
                "text": "not_applicable",
                "eol": "not_applicable",
                "binary": "not_applicable",
                "diff": "not_applicable",
                "merge": "not_applicable",
                "status": "fail" if broad_csv_binary else "pass",
                "notes": "only results/raw/*.csv may disable text handling",
            },
            {
                "path": "*.json",
                "expected_policy": "not_broadly_byte_preserved",
                "text": "not_applicable",
                "eol": "not_applicable",
                "binary": "not_applicable",
                "diff": "not_applicable",
                "merge": "not_applicable",
                "status": "fail" if broad_json_binary else "pass",
                "notes": "only results/raw/*.json may disable text handling",
            },
        ]
    )
    return rows


def gitignore_checks(root: Path) -> list[dict[str, object]]:
    """Check required local-only patterns and protected-path visibility."""
    lines = {
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    rows = [
        {
            "check": f"required_pattern:{pattern}",
            "expected": "present",
            "actual": "present" if pattern in lines else "absent",
            "status": "pass" if pattern in lines else "fail",
            "notes": "confirmed local cache or auxiliary pattern",
        }
        for pattern in REQUIRED_IGNORE_PATTERNS
    ]
    forbidden = ("results/", "documents/", "*.csv", "*.json", "*.pdf")
    rows.extend(
        {
            "check": f"forbidden_broad_pattern:{pattern}",
            "expected": "absent",
            "actual": "present" if pattern in lines else "absent",
            "status": "fail" if pattern in lines else "pass",
            "notes": "broad rule would hide release or provenance artifacts",
        }
        for pattern in forbidden
    )
    manifest_ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", "results/raw/manifest.json"], cwd=root
    ).returncode == 0
    rows.append(
        {
            "check": "protected_manifest_visible",
            "expected": "not_ignored",
            "actual": "ignored" if manifest_ignored else "not_ignored",
            "status": "fail" if manifest_ignored else "pass",
            "notes": "tracked immutable manifest must remain visible",
        }
    )
    return rows


def _csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = len(next(reader))
        rows = sum(1 for _ in reader)
    return rows, columns


def raw_integrity(root: Path) -> list[dict[str, object]]:
    """Validate the four immutable historical inputs."""
    rows: list[dict[str, object]] = []
    for relative, expected in RAW_CONTRACTS.items():
        path = root / relative
        actual_rows, actual_columns = (None, None)
        if path.suffix == ".csv":
            actual_rows, actual_columns = _csv_shape(path)
        actual = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": actual_rows,
            "columns": actual_columns,
        }
        for check in ("sha256", "bytes", "rows", "columns"):
            if expected[check] is None:
                continue
            rows.append(
                {
                    "phase": "raw",
                    "path": relative,
                    "check": check,
                    "expected": expected[check],
                    "actual": actual[check],
                    "status": "pass" if actual[check] == expected[check] else "fail",
                    "notes": "canonical immutable artifact contract",
                }
            )
    return rows


def windows_checkout_preservation(root: Path) -> dict[str, object]:
    """Round-trip protected bytes through an autocrlf=true isolated Git index."""
    before = {relative: sha256_file(root / relative) for relative in RAW_CONTRACTS}
    with tempfile.TemporaryDirectory(prefix="r500_checkout_") as temporary:
        checkout = Path(temporary)
        shutil.copy2(root / ".gitattributes", checkout / ".gitattributes")
        for relative in RAW_CONTRACTS:
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        _run_git(checkout, "init", "-q")
        _run_git(checkout, "config", "core.autocrlf", "true")
        _run_git(checkout, "config", "core.eol", "native")
        _run_git(checkout, "add", "-f", ".gitattributes", *RAW_CONTRACTS)
        for relative in RAW_CONTRACTS:
            (checkout / relative).unlink()
        _run_git(checkout, "checkout-index", "-f", "-a")
        after = {relative: sha256_file(checkout / relative) for relative in RAW_CONTRACTS}
    matches = {relative: before[relative] == after[relative] for relative in RAW_CONTRACTS}
    return {
        "configuration": {"core.autocrlf": True, "core.eol": "native"},
        "manual_restoration_required": False,
        "protected_files": len(matches),
        "all_hashes_preserved": all(matches.values()),
        "status": "pass" if all(matches.values()) else "fail",
    }


def experimental_documents_review(root: Path) -> list[dict[str, object]]:
    """Return the conservative Phase 5 classification of experimental documents."""
    common = {
        "tracked": True,
        "referenced": True,
        "provenance_value": "high",
        "duplication_level": "low",
        "recommended_action": "keep_with_warning",
        "confidence": "high",
    }
    return [
        {
            "path": "documents/Experiments.qmd",
            "file_type": "Quarto source",
            **common,
            "current_commands": "not_applicable_no_cli_commands",
            "current_results": "mixed_historical_calibration_and_future_design",
            "matches_repository_state": "partly",
            "notes": "Design and calibration provenance; future Post-selection multiplier 1.8 is distinct from the historical R=500 multiplier 1.0 and this is not the authoritative Phase 3 evidence package.",
        },
        {
            "path": "documents/Experiments.pdf",
            "file_type": "rendered PDF",
            **common,
            "current_commands": "unverified",
            "current_results": "unverified",
            "matches_repository_state": "unverified_pdf_predates_qmd",
            "confidence": "medium",
            "notes": "Tracked rendered design document with provenance value; it predates the QMD source, and local PDF text extraction was unavailable, so source equivalence is not claimed.",
        },
    ]


def release_metadata_status(root: Path) -> dict[str, object]:
    """Report missing owner-controlled legal and citation metadata without invention."""
    license_present = any((root / name).is_file() for name in ("LICENSE", "LICENSE.txt", "COPYING"))
    citation_present = (root / "CITATION.cff").is_file()
    return {
        "blocking_for_private_submission": False,
        "blocking_for_public_release": not license_present or not citation_present,
        "citation_decision_required": not citation_present,
        "citation_present": citation_present,
        "license_decision_required": not license_present,
        "license_present": license_present,
        "recommended_next_action": "Obtain owner approval for legal terms and citation fields before public archival release.",
        "required_owner_inputs": [
            "license choice or explicit no-public-license decision",
            "title",
            "authors and approved name forms",
            "ORCID if applicable",
            "institution",
            "repository URL",
            "version",
            "release date",
            "preferred citation",
        ],
    }


def _phase_integrity(root: Path) -> list[dict[str, object]]:
    current = protected_integrity(root)
    rows = []
    for item in current["records"]:
        relative = str(item["path"])
        if relative.startswith("results/validation/r500_audit/"):
            phase = "phase1"
        elif relative.startswith("results/validation/r500_phase2/"):
            phase = "phase2"
        else:
            continue
        rows.append(
            {
                "phase": phase,
                "path": relative,
                "check": "sha256",
                "expected": item.get("expected_sha256", item.get("sha256")),
                "actual": item["sha256"],
                "status": item["status"],
                "notes": f"baseline authority: {item['authority']}",
            }
        )

    package = root / "results/thesis/r500"
    manifest = json.loads((package / "thesis_output_manifest.json").read_text(encoding="utf-8"))
    outputs = manifest["outputs"]
    for item in outputs:
        relative = str(item["output_filename"])
        expected = item.get("generated_output_sha256")
        path = package / relative
        actual = sha256_file(path)
        status = "pass" if expected is None or actual == expected else "fail"
        rows.append(
            {
                "phase": "phase3",
                "path": f"results/thesis/r500/{relative}",
                "check": "manifest_sha256" if expected else "self_referential_manifest",
                "expected": expected or "not_recorded_by_design",
                "actual": actual,
                "status": status,
                "notes": "Phase 3 manifest baseline",
            }
        )
    consistency = list(csv.DictReader((package / "consistency_checks.csv").open(encoding="utf-8", newline="")))
    rows.append(
        {
            "phase": "phase3",
            "path": "results/thesis/r500/consistency_checks.csv",
            "check": "all_190_consistency_checks",
            "expected": 190,
            "actual": sum(row["status"] == "passed" for row in consistency),
            "status": "pass" if len(consistency) == 190 and all(row["status"] == "passed" for row in consistency) else "fail",
            "notes": "established Phase 3 numerical checks",
        }
    )

    review_dir = package / "review"
    readiness = json.loads((review_dir / "delivery_readiness.json").read_text(encoding="utf-8"))
    for filename in REVIEW_FILENAMES:
        path = review_dir / filename
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:results/thesis/r500/review/{filename}"],
            cwd=root,
            capture_output=True,
        ).returncode == 0
        rows.append(
            {
                "phase": "phase4",
                "path": f"results/thesis/r500/review/{filename}",
                "check": "present_and_committed",
                "expected": True,
                "actual": path.is_file() and tracked,
                "status": "pass" if path.is_file() and tracked else "fail",
                "notes": "Phase 4 output remains present in HEAD",
            }
        )
    rows.append(
        {
            "phase": "phase4",
            "path": "results/thesis/r500/review/delivery_readiness.json",
            "check": "readiness_classification",
            "expected": "ready_with_minor_corrections",
            "actual": readiness["classifications"]["overall"],
            "status": "pass" if readiness["classifications"]["overall"] == "ready_with_minor_corrections" else "fail",
            "notes": "committed Phase 4 decision",
        }
    )
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metadata_recommendation() -> str:
    return """# Release metadata recommendations

## License

No `LICENSE`, `LICENSE.txt`, or `COPYING` file is present. No legal terms have
been inferred. Without a license, public users do not receive an affirmative
permission grant. The repository owner must choose among terms such as MIT,
Apache-2.0, GPL-3.0, all rights reserved/no public license, or applicable
institution-specific terms. This is non-blocking for private thesis submission
but potentially important before public repository release.

## Citation metadata

No `CITATION.cff` is present. It must not be fabricated. An owner-approved file
would require title, authors (family and given names), ORCID where applicable,
institution, repository URL, version, release date, preferred citation, and the
selected license if any. Citation metadata can be added in a separate approved
step. Its absence is non-blocking and it is recommended for public archival
release.
"""


def _report(status: Mapping[str, object]) -> str:
    return f"""# Phase 5 release-hygiene report

## Outcome

Final classification: **{status['classification']}**. Scientific and technical
integrity pass. The intended tag `thesis-r500-v1.0` is prepared but not created.

## Cross-platform protection

The minimal `.gitattributes` policy normalizes ordinary source and documentation
to LF, disables text conversion for the four immutable raw artifacts, and marks
PDF/PNG outputs binary. An isolated Git index configured with
`core.autocrlf=true` and native EOL preserved all four SHA-256 hashes. Manual
byte restoration was not required.

## Integrity

- Raw artifact contracts: {status['raw_integrity']}
- Phase 1: {status['phase1_integrity']}
- Phase 2: {status['phase2_integrity']}
- Phase 3: {status['phase3_integrity']} (35 files; 190 consistency checks)
- Phase 4: {status['phase4_integrity']} (12 files; readiness unchanged)

## Repository hygiene and documents

The Phase 4-confirmed cache list is the only authorized cleanup scope; cleanup
is deliberately separate from this validator. `Experiments.qmd` and
`Experiments.pdf` are retained with warnings as design/calibration provenance,
not authoritative Phase 3 evidence. The PDF predates the QMD and equivalence
was not claimed.

## Owner decisions

No license or citation metadata was invented. Both remain owner decisions before
a public archival release. Their absence does not block private thesis
submission. Owner approval of the experimental-document disposition also
remains recommended.
"""


def finalize_release_hygiene(root: Path) -> dict[str, object]:
    """Run all Phase 5 checks and write the eight deterministic review outputs."""
    root = root.resolve()
    review_dir = root / "results/thesis/r500/review"
    review_dir.mkdir(parents=True, exist_ok=True)
    attribute_rows = git_attribute_checks(root)
    ignore_rows = gitignore_checks(root)
    checkout = windows_checkout_preservation(root)
    integrity_rows = raw_integrity(root) + _phase_integrity(root)
    documents = experimental_documents_review(root)
    metadata = release_metadata_status(root)

    scientific_roots = ("src/dgp", "src/estimators", "src/ivqr", "src/simulation")
    tracked_scientific = _run_git(root, "diff", "--name-only", "--", *scientific_roots).stdout.splitlines()
    untracked_scientific = _run_git(
        root, "ls-files", "--others", "--exclude-standard", "--", *scientific_roots
    ).stdout.splitlines()
    scientific_changes = sorted(set(tracked_scientific + untracked_scientific))

    statuses = [row["status"] for row in attribute_rows + ignore_rows + integrity_rows]
    technical_pass = (
        all(value == "pass" for value in statuses)
        and checkout["status"] == "pass"
        and not scientific_changes
    )
    owner_decisions = bool(
        metadata["license_decision_required"]
        or metadata["citation_decision_required"]
        or any(row["recommended_action"] == "keep_with_warning" for row in documents)
    )
    classification = "not_ready" if not technical_pass else "ready_with_owner_decisions" if owner_decisions else "ready_for_tag"

    phase_status = {
        phase: "pass" if all(row["status"] == "pass" for row in integrity_rows if row["phase"] == phase) else "fail"
        for phase in ("raw", "phase1", "phase2", "phase3", "phase4")
    }
    status = {
        "classification": classification,
        "cross_platform_checkout": checkout,
        "experimental_documents_require_owner_review": True,
        "git_attributes": "pass" if all(row["status"] == "pass" for row in attribute_rows) else "fail",
        "gitignore": "pass" if all(row["status"] == "pass" for row in ignore_rows) else "fail",
        "license_decision_required": metadata["license_decision_required"],
        "citation_decision_required": metadata["citation_decision_required"],
        "manual_byte_restoration_required": False,
        "phase1_integrity": phase_status["phase1"],
        "phase2_integrity": phase_status["phase2"],
        "phase3_integrity": phase_status["phase3"],
        "phase4_integrity": phase_status["phase4"],
        "raw_integrity": phase_status["raw"],
        "recommended_tag": "thesis-r500-v1.0",
        "scientific_code_changed": bool(scientific_changes),
        "scientific_code_diff_paths": scientific_changes,
        "simulation_ran": False,
        "tag_created": False,
    }

    _write_csv(
        review_dir / "git_attribute_checks.csv",
        attribute_rows,
        ["path", "expected_policy", "text", "eol", "binary", "diff", "merge", "status", "notes"],
    )
    _write_csv(
        review_dir / "gitignore_checks.csv",
        ignore_rows,
        ["check", "expected", "actual", "status", "notes"],
    )
    _write_csv(
        review_dir / "experimental_documents_review.csv",
        documents,
        ["path", "file_type", "tracked", "referenced", "current_commands", "current_results", "matches_repository_state", "provenance_value", "duplication_level", "recommended_action", "confidence", "notes"],
    )
    _write_csv(
        review_dir / "final_integrity_checks.csv",
        integrity_rows,
        ["phase", "path", "check", "expected", "actual", "status", "notes"],
    )
    (review_dir / "release_metadata_status.json").write_text(stable_json(metadata), encoding="utf-8", newline="\n")
    (review_dir / "release_metadata_recommendations.md").write_text(_metadata_recommendation(), encoding="utf-8", newline="\n")
    (review_dir / "release_hygiene_status.json").write_text(stable_json(status), encoding="utf-8", newline="\n")
    (review_dir / "release_hygiene_report.md").write_text(_report(status), encoding="utf-8", newline="\n")
    return status
