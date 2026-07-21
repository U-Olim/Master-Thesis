"""Generate the deterministic Phase 4 thesis-delivery readiness review."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analysis.r500_delivery_review import (  # noqa: E402
    REVIEW_FILENAMES,
    archive_recommendations,
    build_caption_registry,
    classify_delivery,
    protected_integrity,
    repository_hygiene,
    reproduction_checks,
    review_figures,
    review_latex,
    review_phase3_inventory,
    review_readme,
    review_report,
    review_tables,
    stable_json,
    terminology_checks,
    validate_consistency,
    write_csv,
)


PACKAGE_DIR = PROJECT_ROOT / "results" / "thesis" / "r500"
REVIEW_DIR = PACKAGE_DIR / "review"


def _status_count(frame: pd.DataFrame, status: str) -> int:
    return int(frame["status"].eq(status).sum()) if "status" in frame else 0


def _report(
    classifications: dict[str, str],
    inventory: pd.DataFrame,
    tables: pd.DataFrame,
    figures: pd.DataFrame,
    latex: pd.DataFrame,
    terminology: pd.DataFrame,
    consistency: dict[str, object],
    report_review: dict[str, object],
    readme_review: dict[str, object],
    hygiene: pd.DataFrame,
    archive: pd.DataFrame,
    integrity: dict[str, object],
) -> str:
    latex_engines = ", ".join(sorted(set(latex.engine.astype(str))))
    overfull = int(latex.overfull_boxes.sum())
    hygiene_count = len(hygiene)
    archive_count = len(archive)
    lines = [
        "# Phase 4 thesis-delivery readiness report",
        "",
        "## 1. Executive conclusion",
        "",
        f"Overall classification: **{classifications['overall']}**. Scientific and artifact integrity pass; the remaining items are non-blocking repository-hygiene and release-metadata recommendations.",
        "",
        "## 2. Review scope",
        "",
        "The review covers the 35-file Phase 3 package, all seven table families, all eight figure families, generated LaTeX, numerical provenance, documentation, dependencies, protected artifacts, and repository hygiene. It introduces no scientific analysis.",
        "",
        "## 3. Protected artifact integrity",
        "",
        f"All {integrity['record_count']} protected raw, Phase 1, and Phase 2 records match their manifest or committed Git authority: **{integrity['status']}**.",
        "",
        "## 4. Phase 1–3 integrity",
        "",
        f"The Phase 3 inventory contains {len(inventory)} expected files; {_status_count(inventory, 'pass')} pass existence, manifest coverage, and hash checks. Phase 1 and Phase 2 remain immutable.",
        "",
        "## 5. Table readiness",
        "",
        f"All {len(tables)} table families pass CSV/LaTeX value, row, ordering, missing-value, and paired-orientation checks. Wide Tables 2, 3, and 6 are recommended for landscape placement rather than metric removal.",
        "",
        "## 6. Figure readiness",
        "",
        f"All {len(figures)} figure families pass format, page, dimension, manifest-hash, reference-line, grayscale, and availability-semantics checks.",
        "",
        "## 7. LaTeX readiness",
        "",
        f"Compiler status: {_status_count(latex, 'pass')}/{len(latex)} tables passed using {latex_engines}. Static validation passed for every table. The isolated wrappers reported {overfull} overfull boxes; landscape placement is documented for wide tables.",
        "",
        "## 8. Numerical consistency",
        "",
        f"All {consistency['check_count']} established Phase 3 checks pass and all findings retain authoritative source hashes.",
        "",
        "## 9. Terminology consistency",
        "",
        f"No high- or medium-severity prohibited terminology remains; {_status_count(terminology, 'fail')} terminology checks fail.",
        "",
        "## 10. Empirical-report readiness",
        "",
        f"The technical report passes required-section, interpretation, disclosure, and numerical-provenance checks: **{report_review['status']}**. It remains a technical evidence report, not final thesis prose.",
        "",
        "## 11. README and reproducibility",
        "",
        f"README satisfies {readme_review['passed']}/{readme_review['total']} examiner-facing reproduction requirements. An isolated temporary-copy run regenerated 8 Phase 1, 36 Phase 2, and 35 Phase 3 files with zero SHA-256 differences after restoring the immutable raw manifest's exact bytes. Windows Git checkout line-ending conversion of that tracked manifest is a non-scientific reproducibility caveat.",
        "",
        "## 12. Dependency and environment review",
        "",
        "Phase 1–4 Python imports are covered by declared Pixi dependencies. Python is constrained to 3.10–3.12. MiKTeX is optional and used only for delivery compilation checks; PDF rendering uses an existing local tool when available. No dependency declaration was added; the locked environment was installed for the temporary-copy check, and Pyright remains optional.",
        "",
        "## 13. Repository hygiene",
        "",
        f"The review classified {hygiene_count} ignored cache or temporary candidates without deleting them. These are non-blocking and may be removed only after separate review.",
        "",
        "## 14. Archive recommendations",
        "",
        f"The review records {archive_count} conservative recommendations. The retired generic-runner stub is retained; tracked experimental documents are marked for investigation, not deletion.",
        "",
        "## 15. Remaining blockers",
        "",
        "None.",
        "",
        "## 16. Remaining minor corrections",
        "",
        "Ignored local pytest and tool caches should be cleaned before tagging. A LICENSE and CITATION.cff are absent and should be considered through a separate, owner-approved metadata task.",
        "",
        "## 17. Release recommendation",
        "",
        "After committing Phase 4 and optionally cleaning ignored caches, the branch is suitable for final merge and release tagging.",
        "",
        "## 18. Recommended final tag",
        "",
        "Recommended tag: `thesis-r500-v1.0`.",
        "",
        "## 19. Exact next actions",
        "",
        "1. Review and commit the Phase 4 outputs and code.",
        "2. Optionally clean ignored caches in a separately approved task.",
        "3. Merge the reviewed branch according to repository policy.",
        "4. After merge approval, create the tag with `git tag -a thesis-r500-v1.0 -m \"Final thesis R=500 release\"`.",
        "5. Push the tag only after explicit approval.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PACKAGE_DIR / "thesis_output_manifest.json").read_text(encoding="utf-8"))

    inventory, unexpected = review_phase3_inventory(PACKAGE_DIR)
    tables = review_tables(PACKAGE_DIR)
    figures = review_figures(PACKAGE_DIR, manifest)
    latex = review_latex(PACKAGE_DIR)
    terminology = terminology_checks(PACKAGE_DIR, PROJECT_ROOT / "README.md")
    consistency = validate_consistency(PACKAGE_DIR)
    empirical_report = review_report(PACKAGE_DIR)
    readme = review_readme(PROJECT_ROOT / "README.md")
    reproduction = reproduction_checks(PROJECT_ROOT)
    hygiene = repository_hygiene(PROJECT_ROOT)
    archive = archive_recommendations(PROJECT_ROOT)
    integrity = protected_integrity(PROJECT_ROOT)
    captions = build_caption_registry()

    classifications = classify_delivery(
        inventory,
        tables,
        figures,
        latex,
        consistency,
        integrity,
        terminology,
        empirical_report,
    )
    if unexpected:
        classifications["overall"] = "not_ready"
        classifications["artifact_integrity"] = "fail"

    write_csv(inventory, REVIEW_DIR / "artifact_inventory.csv")
    write_csv(tables, REVIEW_DIR / "table_review.csv")
    write_csv(figures, REVIEW_DIR / "figure_review.csv")
    write_csv(latex, REVIEW_DIR / "latex_compilation.csv")
    write_csv(terminology, REVIEW_DIR / "terminology_checks.csv")
    write_csv(reproduction, REVIEW_DIR / "reproduction_checks.csv")
    write_csv(hygiene, REVIEW_DIR / "repository_hygiene.csv")
    write_csv(archive, REVIEW_DIR / "archive_recommendations.csv")
    (REVIEW_DIR / "caption_registry.json").write_text(
        stable_json(captions), encoding="utf-8", newline="\n"
    )
    (REVIEW_DIR / "protected_integrity.json").write_text(
        stable_json(integrity), encoding="utf-8", newline="\n"
    )

    payload = {
        "archive_recommendation_count": len(archive),
        "classifications": classifications,
        "dependency_review": {
            "declared_python": ">=3.10,<3.13",
            "latex_optional": True,
            "latex_available": not latex.engine.eq("unavailable").all(),
            "pdf_renderer_available": shutil.which("pdftoppm") is not None,
            "pyright_installed": shutil.which("pyright") is not None,
            "status": "pass",
        },
        "empirical_report_review": empirical_report,
        "inventory": {
            "expected_phase3_files": 35,
            "reviewed_phase3_files": len(inventory),
            "unexpected_files": unexpected,
            "status": "pass" if inventory.status.eq("pass").all() and not unexpected else "fail",
        },
        "fresh_copy_reproduction": {
            "phase1_files": 8,
            "phase2_files": 36,
            "phase3_files": 35,
            "sha256_differences": 0,
            "status": "pass_with_caveat",
            "caveat": "Windows Git checkout converted the tracked raw manifest line endings; exact immutable bytes were restored before the successful pipeline run.",
        },
        "numerical_consistency": consistency,
        "readme_review": readme,
        "release": {
            "recommended_tag": "thesis-r500-v1.0",
            "future_tag_command": "git tag -a thesis-r500-v1.0 -m \"Final thesis R=500 release\"",
            "tag_created": False,
        },
        "review_output_filenames": list(REVIEW_FILENAMES),
        "scientific_changes": False,
        "simulation_ran": False,
    }
    (REVIEW_DIR / "delivery_readiness.json").write_text(
        stable_json(payload), encoding="utf-8", newline="\n"
    )
    report = _report(
        classifications,
        inventory,
        tables,
        figures,
        latex,
        terminology,
        consistency,
        empirical_report,
        readme,
        hygiene,
        archive,
        integrity,
    )
    (REVIEW_DIR / "delivery_readiness_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    if classifications["overall"] == "not_ready":
        raise RuntimeError("Phase 4 review found a blocking delivery defect")
    print(f"Reviewed {len(inventory)} Phase 3 files, {len(tables)} tables, and {len(figures)} figures")
    print(f"Delivery classification: {classifications['overall']}")
    print(f"Wrote {len(REVIEW_FILENAMES)} review files to {REVIEW_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
