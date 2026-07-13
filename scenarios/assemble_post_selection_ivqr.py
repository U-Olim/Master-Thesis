"""Validate and assemble the five Post-selection IVQR R=500 result blocks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from simulation.post_selection_output import clean_post_selection_results_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "raw" / "post_selection_ivqr"
OUTPUT_CSV = RESULT_DIR / "post_selection_ivqr_full.csv"
OUTPUT_XLSX = RESULT_DIR / "post_selection_ivqr_full.xlsx"
CLEAN_OUTPUT_CSV = (
    REPO_ROOT
    / "results"
    / "clean"
    / "post_selection_ivqr"
    / "post_selection_ivqr_full.csv"
)

EXPECTED_BLOCKS = (
    ("post_selection_R500_lasso180_grid21_block000_099.csv", 0, 99),
    ("post_selection_R500_lasso180_grid21_block100_199.csv", 100, 199),
    ("post_selection_R500_lasso180_grid21_block200_299.csv", 200, 299),
    ("post_selection_R500_lasso180_grid21_block300_399.csv", 300, 399),
    ("post_selection_R500_lasso180_grid21_block400_499.csv", 400, 499),
)

# These are the actual design and replication column names in the source files.
DESIGN_COLUMNS = ["dgp", "n", "p", "pi", "tau"]
REP_COLUMN = "rep"
NATURAL_KEY = [*DESIGN_COLUMNS, REP_COLUMN]
EXPECTED_REPLICATIONS = set(range(500))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excel_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to values accepted by XlsxWriter."""
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if item is not None else value


def _write_formatted_xlsx(data: pd.DataFrame, path: Path) -> None:
    try:
        import xlsxwriter
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "XLSX output requires XlsxWriter. Install it or run with --no-xlsx."
        ) from exc

    workbook = xlsxwriter.Workbook(
        path,
        {"constant_memory": True, "strings_to_urls": False, "nan_inf_to_errors": False},
    )
    try:
        worksheet = workbook.add_worksheet("Post-selection IVQR")
        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#4F81BD",
                "font_color": "#FFFFFF",
                "border": 1,
            }
        )
        worksheet.write_row(0, 0, list(data.columns), header_format)
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(data), len(data.columns) - 1)

        for row_number, row in enumerate(
            data.itertuples(index=False, name=None), start=1
        ):
            worksheet.write_row(row_number, 0, [_excel_value(value) for value in row])
    finally:
        workbook.close()


def assemble(*, write_xlsx: bool = True) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    block_paths = [RESULT_DIR / name for name, _, _ in EXPECTED_BLOCKS]
    missing_files = [path for path in block_paths if not path.is_file()]
    if missing_files:
        missing = "\n  ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required block file(s):\n  {missing}")

    resolved_paths = [path.resolve() for path in block_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("A source block path was included more than once.")

    hashes = [_sha256(path) for path in block_paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Two source blocks have identical content.")

    frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    source_rows: list[tuple[Path, int]] = []
    reference_columns: list[str] | None = None

    for (name, lower, upper), path in zip(EXPECTED_BLOCKS, block_paths, strict=True):
        frame = pd.read_csv(path, low_memory=False)
        # Keep a text-preserving copy so CSV output never round-trips numeric
        # fields through binary floating point.
        raw_frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
        columns = list(frame.columns)
        if reference_columns is None:
            reference_columns = columns
            missing_columns = [
                column for column in NATURAL_KEY if column not in reference_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Required natural-key column(s) missing: {missing_columns}"
                )
        elif columns != reference_columns:
            raise ValueError(
                f"Column names or order in {name} differ from the first block."
            )
        if list(raw_frame.columns) != columns:
            raise ValueError(f"Text-preserving read changed the schema of {name}.")

        actual_reps = set(frame[REP_COLUMN].unique())
        expected_reps = set(range(lower, upper + 1))
        if actual_reps != expected_reps:
            missing = sorted(expected_reps - actual_reps)
            unexpected = sorted(actual_reps - expected_reps)
            raise ValueError(
                f"Replication range mismatch in {name}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        frames.append(frame)
        raw_frames.append(raw_frame)
        source_rows.append((path, len(frame)))

    assert reference_columns is not None
    combined = pd.concat(frames, axis=0, ignore_index=True)
    raw_combined = pd.concat(raw_frames, axis=0, ignore_index=True)

    actual_reps = set(combined[REP_COLUMN].unique())
    missing_reps = sorted(EXPECTED_REPLICATIONS - actual_reps)
    unexpected_reps = sorted(actual_reps - EXPECTED_REPLICATIONS)
    if missing_reps or unexpected_reps:
        raise ValueError(
            "Combined replication coverage is not 0 through 499: "
            f"missing={missing_reps}, unexpected={unexpected_reps}"
        )
    if combined[REP_COLUMN].nunique() != 500:
        raise ValueError("The combined data do not contain exactly 500 replications.")

    duplicate_count = int(combined.duplicated(subset=NATURAL_KEY).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate row(s) by natural key {NATURAL_KEY}."
        )

    design_rep_counts = combined.groupby(
        DESIGN_COLUMNS, dropna=False, sort=False
    )[REP_COLUMN].nunique()
    if not (design_rep_counts == 500).all():
        bad_counts = design_rep_counts[design_rep_counts != 500]
        raise ValueError(
            "Not every design cell has 500 unique replications:\n"
            f"{bad_counts.to_string()}"
        )

    sort_order = combined.sort_values(NATURAL_KEY, kind="mergesort").index
    combined = combined.loc[sort_order].reset_index(drop=True)
    raw_combined = raw_combined.loc[sort_order].reset_index(drop=True)
    raw_combined.to_csv(OUTPUT_CSV, index=False)
    clean_post_selection_results_csv(OUTPUT_CSV, CLEAN_OUTPUT_CSV)

    if write_xlsx:
        _write_formatted_xlsx(combined, OUTPUT_XLSX)

    # Required read-back integrity check of the primary CSV output.
    reread = pd.read_csv(OUTPUT_CSV, low_memory=False)
    raw_reread = pd.read_csv(
        OUTPUT_CSV,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    if reread.shape != combined.shape:
        raise ValueError(
            f"CSV read-back shape changed: wrote {combined.shape}, read {reread.shape}."
        )
    if list(reread.columns) != reference_columns:
        raise ValueError("CSV read-back column names or order changed.")
    if not raw_reread.equals(raw_combined):
        raise ValueError("CSV read-back changed one or more source field values.")
    accidental_index_columns = [
        column for column in reread.columns if column.startswith("Unnamed:")
    ]
    if accidental_index_columns:
        raise ValueError(
            f"Accidental index column(s) found: {accidental_index_columns}"
        )

    design_cells = len(design_rep_counts)
    reps_min = int(design_rep_counts.min())
    reps_max = int(design_rep_counts.max())
    discrepancy_messages = []
    if len(combined) != 72_000:
        discrepancy_messages.append(f"rows: expected about 72,000, got {len(combined):,}")
    if len(combined.columns) != 119:
        discrepancy_messages.append(
            f"columns: expected about 119, got {len(combined.columns)}"
        )
    if design_cells != 144:
        discrepancy_messages.append(
            f"design cells: expected about 144, got {design_cells}"
        )

    print("Post-selection IVQR assembly verification")
    print("Source files used:")
    for path, rows in source_rows:
        print(f"  - {path.relative_to(REPO_ROOT)}: {rows:,} rows")
    print(f"Total rows: {len(combined):,}")
    print(f"Total columns: {len(combined.columns)}")
    print(
        f"Replication minimum/maximum: "
        f"{int(combined[REP_COLUMN].min())}/{int(combined[REP_COLUMN].max())}"
    )
    print(f"Unique replications: {combined[REP_COLUMN].nunique()}")
    print(f"Unique design cells ({', '.join(DESIGN_COLUMNS)}): {design_cells}")
    print(f"Replications per design (min/max): {reps_min}/{reps_max}")
    print(f"Duplicate count by natural key ({', '.join(NATURAL_KEY)}): {duplicate_count}")
    print(f"Missing replication count: {len(missing_reps)}")
    print(f"Final CSV: {OUTPUT_CSV}")
    print(f"Clean thesis CSV: {CLEAN_OUTPUT_CSV}")
    if write_xlsx:
        print(f"Formatted XLSX: {OUTPUT_XLSX}")
    print(
        "CSV read-back integrity: PASS "
        f"({reread.shape[0]:,} rows, {reread.shape[1]} columns, "
        "columns and source field text preserved, no index column)"
    )
    if discrepancy_messages:
        print("Expected-size discrepancies:")
        for message in discrepancy_messages:
            print(f"  - {message}")
    else:
        print("Expected-size checks: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip the optional formatted Excel copy.",
    )
    args = parser.parse_args()
    assemble(write_xlsx=not args.no_xlsx)


if __name__ == "__main__":
    main()
