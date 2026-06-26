"""Shared helpers for scenario scripts."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from reporting.figures import write_figures
from reporting.summaries import aggregate_results_file
from reporting.tables import write_tables
from simulation._validation import validate_output_file_path


def validate_output_path(output_path: Path, *, resume: bool) -> None:
    """Validate a scenario output path and protect existing files."""
    validated = validate_output_file_path(output_path)
    if validated.exists() and not resume:
        raise FileExistsError(
            f"Output file already exists: {validated}. "
            "Use --resume to continue, pass --output to choose a new file, "
            "or delete the existing file manually."
        )


def validate_resume_manifest(
    manifest_path: str | Path | None,
    current_signature: dict[str, object],
) -> None:
    """Reject resume attempts whose manifest signature differs from current args."""
    if manifest_path is None:
        return
    path = Path(manifest_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    previous = payload.get("resume_signature")
    if previous is not None and previous != current_signature:
        raise ValueError(
            "Manifest resume signature does not match current run settings. "
            "Use a different output/manifest path or rerun from scratch."
        )


def make_reports(args: Any) -> None:
    """Write summary, table, and figure outputs for a completed scenario run."""
    summary = aggregate_results_file(
        args.output,
        args.summary_output,
        expected_replications=args.reps,
    )
    tables = write_tables(summary, args.tables_dir)
    figures = write_figures(summary, args.figures_dir)
    print(f"Summary: {args.summary_output}")
    for name, path in tables.items():
        print(f"Table ({name}): {path}")
    for name, path in figures.items():
        print(f"Figure ({name}): {path}")


def alpha_grid_step(alpha_min: float, alpha_max: float, alpha_grid_size: int) -> float:
    """Return the implied step for an evenly spaced alpha grid."""
    return (alpha_max - alpha_min) / (alpha_grid_size - 1)


def print_dry_run_common(
    *,
    mode: str,
    number_of_designs: int,
    reps: int,
    alpha_min: float,
    alpha_max: float,
    alpha_grid_size: int,
    output: Path | str,
    resume: bool,
    extra_lines: tuple[str, ...] = (),
) -> None:
    """Print the dry-run fields shared by scenario scripts."""
    print(f"Mode: {mode}")
    print(f"Designs: {number_of_designs}")
    print(f"Replications per design: {reps}")
    print(f"alpha_min = {alpha_min}")
    print(f"alpha_max = {alpha_max}")
    print(f"alpha_grid_size = {alpha_grid_size}")
    print(f"alpha_grid_step = {alpha_grid_step(alpha_min, alpha_max, alpha_grid_size):g}")
    print(f"Output: {output}")
    print(f"Resume: {str(resume).lower()}")
    for line in extra_lines:
        print(line)
    print("Reports: automatic after successful run")
