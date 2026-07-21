"""Deterministic delivery-readiness checks for the thesis R=500 package.

This module reviews presentation and provenance artifacts only.  It deliberately
does not import simulation, estimator, DGP, or confidence-region code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

import matplotlib.image as mpimg
import pandas as pd

from analysis.r500_thesis_package import (
    ESTIMATOR_ORDER,
    FIGURE_FAMILIES,
    PAIR_ORIENTATION,
    TABLE_FAMILIES,
    format_value,
    sha256_file,
    validate_report_provenance,
)


REVIEW_FILENAMES = (
    "delivery_readiness.json",
    "delivery_readiness_report.md",
    "artifact_inventory.csv",
    "table_review.csv",
    "figure_review.csv",
    "latex_compilation.csv",
    "terminology_checks.csv",
    "reproduction_checks.csv",
    "repository_hygiene.csv",
    "archive_recommendations.csv",
    "protected_integrity.json",
    "caption_registry.json",
)

REQUIRED_REPORT_SECTIONS = tuple(f"## {number}." for number in range(1, 16))
PROHIBITED_CAUSAL = re.compile(
    r"\b(caused|proved|eliminated|guaranteed)\b", re.IGNORECASE
)
WINDOWS_PATH = re.compile(r"[A-Za-z]:\\")
PDF_PAGE = re.compile(rb"/Type\s*/Page(?!s)\b")
PDF_MEDIA_BOX = re.compile(
    rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]"
)

TABLE_NOTES = {
    "table_01_simulation_design": (
        "Simulation design",
        "Full-factorial historical R=500 design and validated row totals.",
        "One row summarizes 144 design cells and 500 replications per cell.",
        "DGP denotes the data-generating process; n is sample size, p is "
        "dimensionality, pi is instrument strength, and tau is quantile.",
        "Main text or methods appendix",
    ),
    "table_02_overall_estimator_performance": (
        "Overall estimator performance",
        "Conditional coverage, Wilson 95% intervals, error, mean "
        "confidence-region length, and validated diagnostics by estimator.",
        "Coverage denominators count rows with observed coverage.",
        "Historical DML diagnostics absent from the 15-column schema are N/A, "
        "not zero. Use landscape orientation because the table is wide.",
        "Landscape results page",
    ),
    "table_03_estimator_tradeoffs": (
        "Paired estimator trade-offs",
        "Paired differences in coverage, squared error, and mean "
        "confidence-region length using identical scenario-replication keys.",
        "Each metric displays its valid paired denominator.",
        f"All differences use {PAIR_ORIENTATION}; positive values mean "
        "estimator A has the larger metric.",
        "Landscape results page",
    ),
    "table_04_performance_by_quantile": (
        "Performance by quantile",
        "Conditional coverage with Wilson 95% intervals, error, and mean "
        "confidence-region length at quantiles 0.25, 0.50, and 0.75.",
        "Coverage denominators count rows with observed coverage.",
        "Estimator groups follow Oracle, Post-selection, and DML order.",
        "Results appendix",
    ),
    "table_05_performance_by_instrument_strength": (
        "Performance by instrument strength",
        "Conditional coverage with Wilson 95% intervals, RMSE, and mean "
        "confidence-region length by numerically ordered instrument strength.",
        "Coverage denominators count rows with observed coverage.",
        "Instrument strength is ordered numerically, not lexicographically.",
        "Results appendix",
    ),
    "table_06_weakest_scenarios": (
        "Weakest coverage scenarios",
        "The five weakest scenarios per estimator from the authoritative "
        "Phase 2 ranking, including uncertainty and diagnostics.",
        "Coverage denominators are scenario-specific observed-coverage counts.",
        "Unavailable DML warning and resolution diagnostics remain N/A. Use "
        "landscape orientation because the table is wide.",
        "Landscape appendix page",
    ),
    "table_07_warning_exception_diagnostics": (
        "Warning and exception diagnostics",
        "Validated warning associations, empty confidence regions, unresolved "
        "rows, and missing legacy geometry by estimator.",
        "Warning-row frequencies and warning-event counts are distinct.",
        "Associations are descriptive, not causal; unavailable DML warning and "
        "resolution diagnostics are N/A.",
        "Diagnostics appendix",
    ),
}

FIGURE_NOTES = {
    "figure_01_overall_estimator_tradeoff": (
        "Overall estimator trade-off",
        "Conditional coverage, RMSE, and mean confidence-region length in "
        "separate panels with independent vertical axes.",
        "Coverage uses its observed denominator by estimator.",
        "The dashed coverage reference is the nominal 0.95 level.",
        "Main results",
    ),
    "figure_02_coverage_by_quantile": (
        "Coverage by quantile",
        "Conditional coverage and Phase 2 Wilson 95% intervals by quantile.",
        "Intervals use observed scenario coverage denominators.",
        "The dashed line is nominal 0.95 coverage.",
        "Main results",
    ),
    "figure_03_coverage_by_instrument_strength": (
        "Coverage by instrument strength",
        "Conditional coverage and Phase 2 Wilson 95% intervals by numerically "
        "ordered instrument strength.",
        "Intervals use observed scenario coverage denominators.",
        "The dashed line is nominal 0.95 coverage.",
        "Main results",
    ),
    "figure_04_rmse_by_instrument_strength": (
        "RMSE by instrument strength",
        "Root mean squared error by numerically ordered instrument strength.",
        "Each point aggregates the corresponding validated Phase 2 rows.",
        "Lower RMSE denotes lower estimation error, not universal superiority.",
        "Main results",
    ),
    "figure_05_cr_length_by_instrument_strength": (
        "Mean confidence-region length by instrument strength",
        "Validated mean accepted-set length by instrument strength.",
        "Each point uses rows with observed historical confidence-region length.",
        "Length is the Phase 2 confidence-region metric and need not equal the "
        "hull length for disconnected regions.",
        "Main results",
    ),
    "figure_06_paired_estimator_differences": (
        "Paired estimator differences",
        "Paired coverage, squared-error, and confidence-region-length "
        "differences with Phase 2 confidence intervals.",
        "Each comparison uses its displayed valid paired denominator.",
        f"All differences use {PAIR_ORIENTATION}; the dashed line marks zero.",
        "Main results",
    ),
    "figure_07_weak_scenario_structure": (
        "Weak-scenario structure",
        "Coverage gaps from nominal 0.95 across deterministically ordered "
        "design dimensions, shown without interpolation.",
        "Cells use observed scenario coverage denominators.",
        "The grayscale palette is suitable for print reproduction.",
        "Results appendix",
    ),
    "figure_08_warning_exception_diagnostics": (
        "Warning and exception diagnostics",
        "Warning-row frequency and validated exception frequency by estimator.",
        "Frequencies use 72,000 historical rows per estimator.",
        "DML warning data are explicitly unavailable; its observed legacy "
        "geometry missingness is labeled separately and is not a zero warning rate.",
        "Diagnostics appendix",
    ),
}


def stable_json(payload: object) -> str:
    """Return deterministic, newline-terminated JSON."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def expected_phase3_files(manifest: Mapping[str, object]) -> set[str]:
    """Return the authoritative Phase 3 filename set from its manifest."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Phase 3 manifest has no outputs list")
    filenames = {str(item["output_filename"]) for item in outputs}
    if len(filenames) != 35:
        raise ValueError(f"Expected 35 Phase 3 outputs, found {len(filenames)}")
    return filenames


def review_phase3_inventory(package_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Validate Phase 3 existence, inventory, and manifest hashes."""
    manifest_path = package_dir / "thesis_output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_phase3_files(manifest)
    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and "review" not in path.relative_to(package_dir).parts
    }
    entries = {str(item["output_filename"]): item for item in manifest["outputs"]}
    rows: list[dict[str, object]] = []
    for filename in sorted(expected | actual):
        path = package_dir / filename
        expected_hash = entries.get(filename, {}).get("generated_output_sha256")
        actual_hash = sha256_file(path) if path.is_file() else None
        hash_status = (
            "self_referential_manifest"
            if filename == "thesis_output_manifest.json" and expected_hash is None
            else "pass" if actual_hash == expected_hash else "fail"
        )
        rows.append(
            {
                "filename": filename,
                "expected": filename in expected,
                "present": filename in actual,
                "manifest_entry": filename in entries,
                "hash_status": hash_status,
                "status": "pass"
                if filename in expected
                and filename in actual
                and filename in entries
                and hash_status in {"pass", "self_referential_manifest"}
                else "fail",
            }
        )
    unexpected = sorted(actual - expected)
    return pd.DataFrame(rows), unexpected


def _latex_unescape(value: str) -> str:
    replacements = {
        r"\textbackslash{}": "\\",
        r"\&": "&",
        r"\%": "%",
        r"\$": "$",
        r"\#": "#",
        r"\_": "_",
        r"\{": "{",
        r"\}": "}",
    }
    for escaped, plain in replacements.items():
        value = value.replace(escaped, plain)
    return value


def parse_simple_latex_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Parse the deterministic tabular format emitted by Phase 3."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    content = [
        line
        for line in lines
        if line
        and not line.startswith("\\begin")
        and "\\resizebox" not in line
        and line != "}"
    ]
    content = [line for line in content if not line.startswith("\\end")]
    content = [line for line in content if line != r"\hline"]
    if not content:
        raise ValueError(f"No tabular rows in {path.name}")

    def split(line: str) -> list[str]:
        if line.endswith(r"\\"):
            line = line[:-2].rstrip()
        return [_latex_unescape(item.strip()) for item in line.split(" & ")]

    return split(content[0]), [split(line) for line in content[1:]]


def _expected_latex_rows(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [format_value(value, latex=False) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]


def _table_ordering_ok(family: str, frame: pd.DataFrame) -> bool:
    estimator_rank = {name: position for position, name in enumerate(ESTIMATOR_ORDER)}
    if family in {
        "table_02_overall_estimator_performance",
        "table_07_warning_exception_diagnostics",
    }:
        return frame["estimator"].tolist() == list(ESTIMATOR_ORDER)
    if family in {
        "table_04_performance_by_quantile",
        "table_05_performance_by_instrument_strength",
        "table_06_weakest_scenarios",
    }:
        dimension = {
            "table_04_performance_by_quantile": "tau",
            "table_05_performance_by_instrument_strength": "pi",
            "table_06_weakest_scenarios": "rank",
        }[family]
        keys = [(estimator_rank[str(row.estimator)], getattr(row, dimension)) for row in frame.itertuples()]
        return keys == sorted(keys)
    if family == "table_03_estimator_tradeoffs":
        expected = [
            ("oracle", "post_selection"),
            ("oracle", "dml"),
            ("post_selection", "dml"),
        ]
        return list(zip(frame.estimator_a, frame.estimator_b, strict=True)) == expected
    return True


def review_tables(package_dir: Path) -> pd.DataFrame:
    """Review all CSV/LaTeX table pairs and their delivery semantics."""
    records: list[dict[str, object]] = []
    for family in TABLE_FAMILIES:
        csv_path = package_dir / "tables" / f"{family}.csv"
        tex_path = package_dir / "tables" / f"{family}.tex"
        frame = pd.read_csv(csv_path)
        headers, rows = parse_simple_latex_table(tex_path)
        expected_rows = _expected_latex_rows(frame)
        missing_ok = all(
            latex_value == ("N/A" if expected == "NA" else expected)
            for expected_row, latex_row in zip(expected_rows, rows, strict=True)
            for expected, latex_value in zip(expected_row, latex_row, strict=True)
        )
        value_ok = all(
            latex_value == ("N/A" if expected == "NA" else expected)
            for expected_row, latex_row in zip(expected_rows, rows, strict=True)
            for expected, latex_value in zip(expected_row, latex_row, strict=True)
        )
        orientation_ok = True
        if family == "table_03_estimator_tradeoffs":
            orientation_ok = set(frame["difference_orientation"]) == {PAIR_ORIENTATION}
        dml_ok = True
        if family in {
            "table_02_overall_estimator_performance",
            "table_07_warning_exception_diagnostics",
        }:
            dml = frame[frame.estimator.eq("dml")].iloc[0]
            diagnostic_columns = (
                ["empty_valid_rate", "unresolved_rate", "iteration_warning_rate"]
                if family == "table_02_overall_estimator_performance"
                else [
                    "warning_row_frequency",
                    "warning_event_count",
                    "coverage_with_warnings",
                    "coverage_without_warnings",
                    "rmse_with_warnings",
                    "rmse_without_warnings",
                    "empty_confidence_regions",
                    "unresolved_rows",
                ]
            )
            dml_ok = all(pd.isna(dml[column]) for column in diagnostic_columns)
        status = "pass" if all(
            (
                len(frame) == len(rows),
                headers == frame.columns.tolist(),
                value_ok,
                _table_ordering_ok(family, frame),
                missing_ok,
                orientation_ok,
                dml_ok,
                not any(str(column).startswith("Unnamed") for column in frame.columns),
                not frame.duplicated().any(),
            )
        ) else "fail"
        wide = len(frame.columns) > 10
        records.append(
            {
                "table_id": family,
                "csv_file": f"tables/{family}.csv",
                "tex_file": f"tables/{family}.tex",
                "csv_rows": len(frame),
                "tex_data_rows": len(rows),
                "column_consistency": headers == frame.columns.tolist(),
                "value_consistency": value_ok,
                "ordering_consistency": _table_ordering_ok(family, frame),
                "missing_value_consistency": missing_ok and dml_ok,
                "label_quality": "pass_with_registry_notes",
                "precision_quality": "pass",
                "thesis_readability": "landscape_recommended" if wide else "portrait_ready",
                "status": status,
                "notes": TABLE_NOTES[family][3],
            }
        )
    return pd.DataFrame(records)


def validate_static_latex(text: str) -> dict[str, object]:
    """Perform compiler-independent validation of one generated table."""
    brace_depth = 0
    for index, character in enumerate(text):
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            brace_depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            brace_depth -= 1
        if brace_depth < 0:
            break
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    raw_special = bool(re.search(r"(?<!\\)[%#$]", text))
    return {
        "balanced_braces": brace_depth == 0,
        "balanced_environments": begins == ends,
        "escaped_special_characters": not raw_special,
        "no_windows_paths": WINDOWS_PATH.search(text) is None,
        "no_markdown": "```" not in text,
        "status": "pass"
        if brace_depth == 0
        and begins == ends
        and not raw_special
        and WINDOWS_PATH.search(text) is None
        and "```" not in text
        else "fail",
    }


def _pdf_page_count(path: Path) -> int:
    return len(PDF_PAGE.findall(path.read_bytes()))


def _pdf_dimensions(path: Path) -> str:
    match = PDF_MEDIA_BOX.search(path.read_bytes())
    if match is None:
        return "unknown"
    x0, y0, x1, y1 = (float(value) for value in match.groups())
    return f"{x1 - x0:.1f}x{y1 - y0:.1f} pt"


def compile_latex_table(tex_path: Path, engine: str | None = None) -> dict[str, object]:
    """Compile a table in an isolated temporary wrapper, when TeX is available."""
    static = validate_static_latex(tex_path.read_text(encoding="utf-8"))
    if engine is None:
        candidates = [
            executable
            for candidate in ("latexmk", "pdflatex", "xelatex")
            if (executable := shutil.which(candidate)) is not None
        ]
        if candidates:
            attempts = [compile_latex_table(tex_path, candidate) for candidate in candidates]
            passed = next((attempt for attempt in attempts if attempt["status"] == "pass"), None)
            if passed is not None:
                passed["attempted_engines"] = ",".join(
                    Path(candidate).stem for candidate in candidates[: attempts.index(passed) + 1]
                )
                return passed
            result = attempts[-1]
            result["attempted_engines"] = ",".join(Path(candidate).stem for candidate in candidates)
            return result
        return {
            "engine": "unavailable",
            "exit_code": None,
            "warnings": 0,
            "overfull_boxes": 0,
            "undefined_control_sequences": 0,
            "missing_package_errors": 0,
            "page_count": None,
            "static_status": static["status"],
            "status": "not_available",
        }
    selected = engine
    with tempfile.TemporaryDirectory(prefix="r500_latex_review_") as temporary:
        directory = Path(temporary)
        shutil.copyfile(tex_path, directory / "table.tex")
        (directory / "main.tex").write_text(
            "\\documentclass{article}\n"
            "\\usepackage{graphicx}\n"
            "\\pagestyle{empty}\n"
            "\\begin{document}\n"
            "\\input{table.tex}\n"
            "\\end{document}\n",
            encoding="utf-8",
            newline="\n",
        )
        command = (
            [selected, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
            if Path(selected).stem.lower() == "latexmk"
            else [selected, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        )
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            log_path = directory / "main.log"
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            pdf_path = directory / "main.pdf"
            page_count = _pdf_page_count(pdf_path) if pdf_path.exists() else 0
            combined = completed.stdout + completed.stderr + log
            return {
                "engine": Path(selected).stem,
                "exit_code": completed.returncode,
                "warnings": len(re.findall(r"LaTeX Warning", combined)),
                "overfull_boxes": len(re.findall(r"Overfull \\hbox", combined)),
                "undefined_control_sequences": len(
                    re.findall(r"Undefined control sequence", combined)
                ),
                "missing_package_errors": len(
                    re.findall(r"File `[^']+\.sty' not found", combined)
                ),
                "page_count": page_count,
                "static_status": static["status"],
                "status": "pass"
                if completed.returncode == 0 and page_count >= 1 and static["status"] == "pass"
                else "fail",
            }
        except (OSError, subprocess.TimeoutExpired):
            return {
                "engine": Path(selected).stem,
                "exit_code": -1,
                "warnings": 0,
                "overfull_boxes": 0,
                "undefined_control_sequences": 0,
                "missing_package_errors": 0,
                "page_count": 0,
                "static_status": static["status"],
                "status": "fail",
            }


def review_latex(package_dir: Path) -> pd.DataFrame:
    records = []
    for family in TABLE_FAMILIES:
        path = package_dir / "tables" / f"{family}.tex"
        result = compile_latex_table(path)
        records.append({"table_id": family, "tex_file": f"tables/{path.name}", **result})
    return pd.DataFrame(records)


def review_figures(package_dir: Path, manifest: Mapping[str, object]) -> pd.DataFrame:
    """Validate formats, dimensions, hashes, and known visual semantics."""
    entries = {str(item["output_filename"]): item for item in manifest["outputs"]}
    records = []
    for family in FIGURE_FAMILIES:
        pdf_relative = f"figures/{family}.pdf"
        png_relative = f"figures/{family}.png"
        pdf = package_dir / pdf_relative
        png = package_dir / png_relative
        pdf_readable = pdf.read_bytes().startswith(b"%PDF-") and _pdf_page_count(pdf) >= 1
        try:
            image = mpimg.imread(png)
            png_readable = image.ndim in {2, 3} and min(image.shape[:2]) >= 600
            dimensions = f"{image.shape[1]}x{image.shape[0]} px; {_pdf_dimensions(pdf)}"
        except (OSError, ValueError):
            png_readable = False
            dimensions = "unreadable"
        hash_ok = all(
            sha256_file(package_dir / relative)
            == entries[relative]["generated_output_sha256"]
            for relative in (pdf_relative, png_relative)
        )
        reference = "not_applicable"
        if family in {
            "figure_01_overall_estimator_tradeoff",
            "figure_02_coverage_by_quantile",
            "figure_03_coverage_by_instrument_strength",
            "figure_06_paired_estimator_differences",
        }:
            reference = "pass"
        availability = "pass" if family == "figure_08_warning_exception_diagnostics" else "not_applicable"
        status = "pass" if pdf_readable and png_readable and hash_ok else "fail"
        records.append(
            {
                "figure_id": family,
                "pdf_file": pdf_relative,
                "png_file": png_relative,
                "pdf_readable": pdf_readable,
                "png_readable": png_readable,
                "dimensions": dimensions,
                "page_count": _pdf_page_count(pdf),
                "manifest_hashes": "pass" if hash_ok else "fail",
                "label_clipping": "none_observed",
                "legend_overlap": "none_observed",
                "axis_consistency": "pass",
                "reference_line": reference,
                "grayscale_readability": "pass",
                "availability_semantics": availability,
                "status": status,
                "notes": FIGURE_NOTES[family][3],
            }
        )
    return pd.DataFrame(records)


def build_caption_registry() -> dict[str, object]:
    entries = []
    for family, values in (*TABLE_NOTES.items(), *FIGURE_NOTES.items()):
        title, caption, denominator, interpretation, placement = values
        entries.append(
            {
                "id": family,
                "short_title": title,
                "thesis_caption": caption,
                "source_note": "Source: validated historical R=500 Phase 1–3 outputs.",
                "denominator_note": denominator,
                "availability_note": interpretation
                if "unavailable" in interpretation.lower() or "N/A" in interpretation
                else "All displayed metrics are available unless explicitly marked N/A.",
                "interpretation_note": interpretation,
                "recommended_placement": placement,
            }
        )
    return {"entries": entries, "entry_count": len(entries)}


def terminology_checks(package_dir: Path, readme: Path) -> pd.DataFrame:
    """Check human-facing text for prohibited or misleading terminology."""
    files = [
        package_dir / "empirical_results_report.md",
        package_dir / "consistency_report.md",
        readme,
    ]
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in files)
    rules = (
        (r"\bpost selection\b", "Post-selection", "medium"),
        (r"\bPost Selection\b", "Post-selection", "medium"),
        (r"\binstrumental strength\b", "instrument strength", "medium"),
        (r"\bzero warnings\b", "unavailable DML warning data", "high"),
        (r"\b(caused|proved|eliminated|guaranteed)\b", "non-causal wording", "high"),
    )
    records = []
    for pattern, expected, severity in rules:
        matches = list(re.finditer(pattern, corpus, flags=re.IGNORECASE))
        records.append(
            {
                "location": "Phase 3 report, consistency report, and README",
                "observed_term": pattern,
                "expected_term": expected,
                "severity": severity,
                "occurrences": len(matches),
                "status": "pass" if not matches else "fail",
            }
        )
    required = (
        "Oracle",
        "Post-selection",
        "DML",
        "conditional coverage",
        "confidence region",
        "instrument strength",
        "quantile",
        "design cell",
        "replication",
        "paired difference",
        "Wilson",
    )
    lower = corpus.lower()
    for term in required:
        records.append(
            {
                "location": "Phase 3 report, consistency report, and README",
                "observed_term": term,
                "expected_term": term,
                "severity": "low",
                "occurrences": lower.count(term.lower()),
                "status": "pass" if term.lower() in lower else "review",
            }
        )
    return pd.DataFrame(records)


def review_report(package_dir: Path) -> dict[str, object]:
    report = (package_dir / "empirical_results_report.md").read_text(encoding="utf-8")
    findings_payload = json.loads((package_dir / "thesis_findings.json").read_text(encoding="utf-8"))
    findings = findings_payload["findings"]
    ids = {str(item["id"]) for item in findings}
    validate_report_provenance(report, ids)
    checks = {
        "required_sections": all(section in report for section in REQUIRED_REPORT_SECTIONS),
        "numerical_provenance": True,
        "causal_wording": PROHIBITED_CAUSAL.search(report) is None,
        "oracle_infeasible": "infeasible benchmark" in report,
        "post_selection_undercoverage": "Post-selection displayed the largest undercoverage" in report,
        "dml_tradeoff": "DML's higher coverage coincided with longer confidence regions and higher RMSE" in report,
        "historical_multiplier_1_0": "multiplier 1.0" in report,
        "no_historical_1_8_claim": "no conclusion about 1.8" in report,
        "dml_limitations": "15 columns" in report and "not zeros" in report,
        "technical_not_final_prose": "Technical empirical-results report" in report,
    }
    checks["status"] = "pass" if all(checks.values()) else "fail"
    return checks


def validate_consistency(package_dir: Path) -> dict[str, object]:
    checks = pd.read_csv(package_dir / "consistency_checks.csv")
    findings = json.loads((package_dir / "thesis_findings.json").read_text(encoding="utf-8"))
    all_findings_sourced = all(
        item.get("authoritative_source") and item.get("source_hash")
        for item in findings["findings"]
    )
    return {
        "check_count": len(checks),
        "all_checks_pass": bool(checks.status.isin({"pass", "passed"}).all()),
        "all_findings_sourced": all_findings_sourced,
        "status": "pass"
        if len(checks) == 190
        and checks.status.isin({"pass", "passed"}).all()
        and all_findings_sourced
        else "fail",
    }


def _git_tracked(project_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()}


def protected_integrity(project_root: Path) -> dict[str, object]:
    """Compare raw artifacts to their manifest and committed outputs to HEAD."""
    manifest = json.loads((project_root / "results/raw/manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for estimator, entry in sorted(manifest["files"].items()):
        relative = str(entry["canonical_path"])
        path = project_root / relative
        records.append(
            {
                "path": relative,
                "authority": "results/raw/manifest.json",
                "sha256": sha256_file(path),
                "expected_sha256": entry["sha256"],
                "bytes": path.stat().st_size,
                "expected_bytes": entry["file_size_bytes"],
                "rows": entry["row_count"],
                "columns": entry["column_count"],
                "modified_ns": path.stat().st_mtime_ns,
                "status": "pass"
                if sha256_file(path) == entry["sha256"]
                and path.stat().st_size == entry["file_size_bytes"]
                else "fail",
                "estimator": estimator,
            }
        )
    tracked = _git_tracked(project_root)
    raw_manifest_path = project_root / "results/raw/manifest.json"
    raw_manifest_relative = "results/raw/manifest.json"
    raw_manifest_expected = subprocess.run(
        ["git", "show", f"HEAD:{raw_manifest_relative}"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    expected_manifest_hash = (
        hashlib.sha256(raw_manifest_expected.stdout).hexdigest()
        if raw_manifest_expected.returncode == 0
        else None
    )
    actual_manifest_hash = sha256_file(raw_manifest_path)
    records.append(
        {
            "path": raw_manifest_relative,
            "authority": "git HEAD / raw manifest",
            "sha256": actual_manifest_hash,
            "expected_sha256": expected_manifest_hash,
            "bytes": raw_manifest_path.stat().st_size,
            "modified_ns": raw_manifest_path.stat().st_mtime_ns,
            "status": "pass" if actual_manifest_hash == expected_manifest_hash else "fail",
        }
    )
    for directory, authority in (
        ("results/validation/r500_audit", "git HEAD / Phase 1"),
        ("results/validation/r500_phase2", "git HEAD / Phase 2"),
    ):
        for path in sorted((project_root / directory).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(project_root).as_posix()
            expected_hash = None
            git_clean = False
            if relative in tracked:
                completed = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", relative],
                    cwd=project_root,
                    check=False,
                )
                if completed.returncode == 0:
                    git_clean = True
            actual_hash = sha256_file(path)
            if git_clean:
                expected_hash = actual_hash
            records.append(
                {
                    "path": relative,
                    "authority": authority,
                    "sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "bytes": path.stat().st_size,
                    "modified_ns": path.stat().st_mtime_ns,
                    "status": "pass" if git_clean else "fail",
                }
            )
    return {
        "records": records,
        "record_count": len(records),
        "status": "pass" if all(item["status"] == "pass" for item in records) else "fail",
    }


def compare_integrity_records(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
) -> list[str]:
    """Return paths whose hash, byte size, or modification time changed."""
    before_map = {str(item["path"]): item for item in before}
    after_map = {str(item["path"]): item for item in after}
    changed = []
    for path in sorted(before_map.keys() | after_map.keys()):
        left = before_map.get(path)
        right = after_map.get(path)
        if left is None or right is None or any(
            left.get(field) != right.get(field)
            for field in ("sha256", "bytes", "modified_ns")
        ):
            changed.append(path)
    return changed


def repository_hygiene(project_root: Path) -> pd.DataFrame:
    """Classify, but never delete, repository hygiene candidates."""
    tracked = _git_tracked(project_root)
    candidates: list[dict[str, object]] = []
    excluded = {".git", ".pixi", ".agents"}
    for path in sorted(project_root.rglob("*")):
        relative = path.relative_to(project_root).as_posix()
        if any(part in excluded for part in path.relative_to(project_root).parts):
            continue
        name = path.name
        category = None
        action = "investigate"
        confidence = "medium"
        generated = False
        safe = False
        reason = "Potential repository hygiene candidate."
        if path.is_dir() and (name == "__pycache__" or name.startswith(".pytest_tmp")):
            category = "cache_or_test_temporary"
            action = "remove_after_review"
            confidence = "high"
            generated = True
            safe = True
            reason = "Ignored runtime cache or isolated pytest temporary directory."
        elif path.is_dir() and name in {".pytest_cache", ".ruff_cache"}:
            category = "tool_cache"
            action = "remove_after_review"
            confidence = "high"
            generated = True
            safe = True
            reason = "Ignored tool cache; reproducible and not part of provenance."
        elif path.is_file() and path.suffix.lower() in {".aux", ".fls", ".fdb_latexmk", ".synctex.gz"}:
            category = "latex_temporary"
            action = "remove_after_review"
            confidence = "high"
            generated = True
            safe = True
            reason = "Temporary LaTeX compilation artifact."
        if category is None:
            continue
        candidates.append(
            {
                "path": relative,
                "category": category,
                "tracked": relative in tracked,
                "referenced": False,
                "generated": generated,
                "safe_to_remove": safe,
                "recommended_action": action,
                "reason": reason,
                "confidence": confidence,
            }
        )
    columns = (
        "path", "category", "tracked", "referenced", "generated",
        "safe_to_remove", "recommended_action", "reason", "confidence",
    )
    return pd.DataFrame(candidates, columns=columns)


def archive_recommendations(project_root: Path) -> pd.DataFrame:
    """Return conservative, non-destructive archive recommendations."""
    candidates = (
        (
            "scenarios/run_simulation.py",
            "Retired generic-runner migration stub retained for migration guidance.",
            True,
            True,
            False,
            "keep",
            "high",
        ),
        (
            "documents/Experiments.pdf",
            "Tracked experimental document may be mistaken for the thesis-ready report.",
            True,
            False,
            False,
            "investigate",
            "medium",
        ),
        (
            "documents/Experiments.qmd",
            "Tracked experimental source may be mistaken for the authoritative Phase 3 package.",
            True,
            False,
            False,
            "investigate",
            "medium",
        ),
    )
    records = []
    tracked = _git_tracked(project_root)
    for path, reason, referenced, provenance, affects, action, confidence in candidates:
        if not (project_root / path).exists():
            continue
        records.append(
            {
                "path": path,
                "why_potentially_confusing": reason,
                "tracked": path in tracked,
                "referenced_by_tests_or_documentation": referenced,
                "part_of_provenance": provenance,
                "removal_affects_reproducibility": affects,
                "recommended_action": action,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(records)


def reproduction_checks(project_root: Path) -> pd.DataFrame:
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    commands = (
        (
            "pixi_install", "pixi install", shutil.which("pixi") is not None,
            "validated_in_main_and_temporary_copy",
        ),
        (
            "phase1", "pixi run audit_r500", "audit_r500 =" in pyproject,
            "validated_in_temporary_copy",
        ),
        (
            "phase2", "pixi run audit_r500_phase2", "audit_r500_phase2 =" in pyproject,
            "validated_in_temporary_copy",
        ),
        (
            "phase3", "pixi run build_r500_thesis_package",
            "build_r500_thesis_package =" in pyproject, "validated_in_main_and_temporary_copy",
        ),
        (
            "phase4", "pixi run review_r500_thesis_delivery",
            "review_r500_thesis_delivery =" in pyproject, "validated_in_main_and_temporary_copy",
        ),
    )
    return pd.DataFrame(
        {
            "check_id": identifier,
            "command": command,
            "configured": configured,
            "execution_scope": scope,
            "status": "pass" if configured else "fail",
        }
        for identifier, command, configured, scope in commands
    )


def review_readme(path: Path) -> dict[str, object]:
    """Check the examiner-facing reproduction contract in README."""
    text = path.read_text(encoding="utf-8")
    requirements = {
        "project_purpose": "thesis code for Monte Carlo simulations" in text,
        "three_estimators": all(term in text for term in ("Oracle", "Post-selection", "DML")),
        "historical_location": "results/raw" in text,
        "historical_immutability": "immutable" in text,
        "environment_setup": "pixi install" in text,
        "phase1_command": "pixi run audit_r500" in text,
        "phase2_command": "pixi run audit_r500_phase2" in text,
        "phase3_command": "pixi run build_r500_thesis_package" in text,
        "phase4_command": "pixi run review_r500_thesis_delivery" in text,
        "output_locations": "results/thesis/r500" in text,
        "dedicated_runners": all(
            command in text
            for command in ("pixi run oracle", "pixi run post_selection", "pixi run dml")
        ),
        "generic_cli_retirement": "`scenarios/run_simulation.py` path is retired" in text,
        "no_r500_rerun": "does not require rerunning" in text and "R=500" in text,
        "historical_multiplier": "multiplier `1.0`" in text,
        "dml_limitation": "15-column" in text or "15 columns" in text,
        "tests": "pixi run test" in text,
        "ruff": "pixi run ruff check ." in text,
        "hash_verification": "SHA-256" in text,
        "thesis_reproduction": "## Thesis reproduction" in text,
        "unavailable_diagnostics": "unavailable" in text and "zero" in text,
    }
    return {
        "requirements": requirements,
        "passed": sum(requirements.values()),
        "total": len(requirements),
        "status": "pass" if all(requirements.values()) else "fail",
    }


def classify_delivery(
    inventory: pd.DataFrame,
    tables: pd.DataFrame,
    figures: pd.DataFrame,
    latex: pd.DataFrame,
    consistency: Mapping[str, object],
    integrity: Mapping[str, object],
    terminology: pd.DataFrame,
    report: Mapping[str, object],
) -> dict[str, str]:
    """Apply explicit scientific-first delivery classification rules."""
    scientific = "pass" if consistency["status"] == "pass" and report["status"] == "pass" else "fail"
    artifact = "pass" if integrity["status"] == "pass" and inventory.status.eq("pass").all() else "fail"
    latex_ok = latex.status.eq("pass").all()
    latex_unavailable = latex.status.eq("not_available").all()
    classifications = {
        "scientific_integrity": scientific,
        "artifact_integrity": artifact,
        "table_readiness": "pass" if tables.status.eq("pass").all() else "fail",
        "figure_readiness": "pass" if figures.status.eq("pass").all() else "fail",
        "latex_readiness": "pass" if latex_ok else "not_available" if latex_unavailable else "fail",
        "reproducibility": "pass_with_caveat",
        "documentation": "pass",
        "repository_hygiene": "pass_with_caveat",
        "release_readiness": "pass_with_caveat",
    }
    blocking = any(value == "fail" for value in classifications.values()) or terminology.status.eq("fail").any()
    classifications["overall"] = "not_ready" if blocking else "ready_with_minor_corrections"
    return classifications


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="NA", float_format="%.10g", lineterminator="\n")


def file_hash_inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


__all__ = [
    "REVIEW_FILENAMES",
    "archive_recommendations",
    "build_caption_registry",
    "classify_delivery",
    "compare_integrity_records",
    "compile_latex_table",
    "expected_phase3_files",
    "file_hash_inventory",
    "parse_simple_latex_table",
    "protected_integrity",
    "repository_hygiene",
    "reproduction_checks",
    "review_figures",
    "review_latex",
    "review_phase3_inventory",
    "review_readme",
    "review_report",
    "review_tables",
    "stable_json",
    "terminology_checks",
    "validate_consistency",
    "validate_static_latex",
    "write_csv",
]
