"""Validate and assemble the five Oracle IVQR R=500 result blocks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from simulation.oracle_output import clean_oracle_results_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "raw" / "oracle_ivqr"
OUTPUT_CSV = RESULT_DIR / "oracle_ivqr_full.csv"
CLEAN_OUTPUT_CSV = (
    REPO_ROOT / "results" / "clean" / "oracle_ivqr" / "oracle_ivqr_full.csv"
)

EXPECTED_BLOCKS = (
    ("oracle_R500_grid21_block000_099.csv", 0, 99),
    ("oracle_R500_grid21_block100_199.csv", 100, 199),
    ("oracle_R500_grid21_block200_299.csv", 200, 299),
    ("oracle_R500_grid21_block300_399.csv", 300, 399),
    ("oracle_R500_grid21_block400_499.csv", 400, 499),
)
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


def assemble() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    block_paths = [RESULT_DIR / name for name, _, _ in EXPECTED_BLOCKS]

    missing_files = [path for path in block_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required block file(s):\n  "
            + "\n  ".join(str(path) for path in missing_files)
        )
    if len({path.resolve() for path in block_paths}) != len(block_paths):
        raise ValueError("A source block path was included more than once.")
    if len({_sha256(path) for path in block_paths}) != len(block_paths):
        raise ValueError("Two source blocks have identical content.")

    frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    source_rows: list[tuple[Path, int]] = []
    reference_columns: list[str] | None = None

    for (name, lower, upper), path in zip(EXPECTED_BLOCKS, block_paths, strict=True):
        frame = pd.read_csv(path, low_memory=False)
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
            missing_key_columns = [
                column for column in NATURAL_KEY if column not in columns
            ]
            if missing_key_columns:
                raise ValueError(
                    f"Missing natural-key column(s): {missing_key_columns}"
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
            raise ValueError(
                f"Replication range mismatch in {name}: "
                f"missing={sorted(expected_reps - actual_reps)}, "
                f"unexpected={sorted(actual_reps - expected_reps)}"
            )

        frames.append(frame)
        raw_frames.append(raw_frame)
        source_rows.append((path, len(frame)))

    assert reference_columns is not None
    combined = pd.concat(frames, ignore_index=True)
    raw_combined = pd.concat(raw_frames, ignore_index=True)

    actual_reps = set(combined[REP_COLUMN].unique())
    missing_reps = sorted(EXPECTED_REPLICATIONS - actual_reps)
    unexpected_reps = sorted(actual_reps - EXPECTED_REPLICATIONS)
    if missing_reps or unexpected_reps:
        raise ValueError(
            "Combined replication coverage is not 0 through 499: "
            f"missing={missing_reps}, unexpected={unexpected_reps}"
        )
    if combined[REP_COLUMN].nunique() != 500:
        raise ValueError("The combined data do not contain 500 replications.")

    duplicate_count = int(combined.duplicated(subset=NATURAL_KEY).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate row(s) by {NATURAL_KEY}."
        )

    design_rep_counts = combined.groupby(
        DESIGN_COLUMNS, dropna=False, sort=False
    )[REP_COLUMN].nunique()
    if not (design_rep_counts == 500).all():
        raise ValueError("Not every design cell has 500 unique replications.")

    sort_order = combined.sort_values(NATURAL_KEY, kind="mergesort").index
    raw_combined = raw_combined.loc[sort_order].reset_index(drop=True)
    raw_combined.to_csv(OUTPUT_CSV, index=False)
    clean_oracle_results_csv(OUTPUT_CSV, CLEAN_OUTPUT_CSV)

    # Read back as text to prove every original field representation survived.
    reread = pd.read_csv(
        OUTPUT_CSV,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    if reread.shape != raw_combined.shape:
        raise ValueError(
            f"CSV read-back shape changed: wrote {raw_combined.shape}, "
            f"read {reread.shape}."
        )
    if list(reread.columns) != reference_columns:
        raise ValueError("CSV read-back column names or order changed.")
    if not reread.equals(raw_combined):
        raise ValueError("CSV read-back changed one or more source field values.")
    accidental_indexes = [
        column for column in reread.columns if column.startswith("Unnamed:")
    ]
    if accidental_indexes:
        raise ValueError(f"Accidental index columns found: {accidental_indexes}")

    print("Oracle IVQR assembly verification")
    print("Source files used:")
    for path, rows in source_rows:
        print(f"  - {path.relative_to(REPO_ROOT)}: {rows:,} rows")
    print(f"Total rows: {len(raw_combined):,}")
    print(f"Total columns: {len(raw_combined.columns)}")
    print(
        f"Replication minimum/maximum: "
        f"{int(combined[REP_COLUMN].min())}/{int(combined[REP_COLUMN].max())}"
    )
    print(f"Unique replications: {combined[REP_COLUMN].nunique()}")
    print(f"Unique design cells: {len(design_rep_counts)}")
    print(
        "Replications per design (min/max): "
        f"{int(design_rep_counts.min())}/{int(design_rep_counts.max())}"
    )
    print(f"Duplicate count by natural key: {duplicate_count}")
    print(f"Missing replication count: {len(missing_reps)}")
    print(f"Final CSV: {OUTPUT_CSV}")
    print(f"Clean thesis CSV: {CLEAN_OUTPUT_CSV}")
    print(
        "CSV read-back integrity: PASS "
        "(shape, columns, source field text, and no index column)"
    )


if __name__ == "__main__":
    assemble()
