"""Regenerate provenance metadata for the three canonical thesis artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analysis.data import (  # noqa: E402
    CURRENT_OUTPUT_COLUMN_COUNTS,
    HISTORICAL_ARTIFACT_COLUMN_COUNTS,
    NATURAL_KEY_COLUMNS,
    RAW_MANIFEST_PATH,
    RAW_MANIFEST_SCHEMA_VERSION,
    RAW_RESULT_FILES,
    sha256_file,
)


EXPECTED_ROWS = 72_000
EXPECTED_REPLICATION_MIN = 0
EXPECTED_REPLICATION_MAX = 499
EXPECTED_UNIQUE_REPLICATIONS = 500
EXPECTED_ROWS_PER_REPLICATION = 144
ARTIFACT_ROLE = "validated_r500_thesis_result"
SOURCE_GIT_REFERENCE = "pre-refactor-r500"
ARTIFACT_SCHEMA_NAMES = {
    "oracle": "historical_oracle_r500_43_column",
    "post_selection": "historical_post_selection_r500_52_column",
    "dml": "historical_dml_r500_15_column",
}
RECORDED_RUN_SETTINGS: dict[str, Any] = {
    "provenance_note": (
        "Settings were recorded by the project before path reconciliation; the exact "
        "commit that created each CSV is not known."
    ),
    "common": {
        "alpha_grid_min": -1.0,
        "alpha_grid_max": 3.0,
        "alpha_grid_size": 21,
        "base_seed": 12345,
        "critical_value_multiplier": 1.0,
    },
    "estimators": {
        "oracle": {},
        "post_selection": {"selection_lasso_multiplier": 1.0},
        "dml": {
            "k_folds": 3,
            "quantile_penalty": 0.07,
            "quantile_solver": "highs-ipm",
            "ridge_alpha": 1.0,
        },
    },
}


def _recorded_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _artifact_entry(estimator: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical raw result does not exist: {path}")
    frame = pd.read_csv(path)
    natural_key = [
        *NATURAL_KEY_COLUMNS,
        *(["estimator"] if "estimator" in frame.columns else []),
    ]
    counts = frame.groupby("rep", dropna=False).size()
    entry: dict[str, Any] = {
        "estimator": estimator,
        "canonical_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_role": ARTIFACT_ROLE,
        "artifact_schema_name": ARTIFACT_SCHEMA_NAMES[estimator],
        "artifact_schema_version": 1,
        "current_code_column_count": CURRENT_OUTPUT_COLUMN_COUNTS[estimator],
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "column_names": list(frame.columns),
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "replication_min": int(frame["rep"].min()),
        "replication_max": int(frame["rep"].max()),
        "unique_replications": int(frame["rep"].nunique()),
        "rows_per_replication": sorted(int(value) for value in counts.unique()),
        "natural_key": natural_key,
        "duplicate_natural_key_count": int(frame.duplicated(natural_key).sum()),
        "created_or_recorded_timestamp": _recorded_timestamp(path),
        "source_git_reference": SOURCE_GIT_REFERENCE,
        "source_git_reference_note": (
            "Known validation reference; not asserted to be the creating commit."
        ),
    }
    _validate_entry(entry)
    return entry


def _validate_entry(entry: dict[str, Any]) -> None:
    estimator = str(entry["estimator"])
    expected = {
        "row_count": EXPECTED_ROWS,
        "column_count": HISTORICAL_ARTIFACT_COLUMN_COUNTS[estimator],
        "replication_min": EXPECTED_REPLICATION_MIN,
        "replication_max": EXPECTED_REPLICATION_MAX,
        "unique_replications": EXPECTED_UNIQUE_REPLICATIONS,
        "rows_per_replication": [EXPECTED_ROWS_PER_REPLICATION],
        "duplicate_natural_key_count": 0,
    }
    for field, expected_value in expected.items():
        if entry[field] != expected_value:
            raise ValueError(
                f"Historical artifact validation failed for {estimator}: {field} "
                f"expected {expected_value!r}, observed {entry[field]!r}"
            )


def build_manifest() -> dict[str, Any]:
    files = {
        estimator: _artifact_entry(estimator, path)
        for estimator, path in RAW_RESULT_FILES.items()
    }
    return {
        "manifest_schema_version": RAW_MANIFEST_SCHEMA_VERSION,
        "artifact_set": "validated_r500_thesis_results",
        "artifact_policy": {
            "immutable": True,
            "role": ARTIFACT_ROLE,
            "statement": (
                "These historical validated artifacts remain authoritative for thesis "
                "analysis and are not automatically rewritten when serializer schemas "
                "evolve. Current code may emit different schemas for future runs."
            ),
        },
        "number_of_files": len(files),
        "total_rows": sum(entry["row_count"] for entry in files.values()),
        "files": files,
        "recorded_run_settings": RECORDED_RUN_SETTINGS,
    }


def write_manifest_atomic(payload: dict[str, Any], path: Path = RAW_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        parsed = json.loads(temporary.read_text(encoding="utf-8"))
        if parsed != payload:
            raise ValueError("Temporary manifest validation failed")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    manifest = build_manifest()
    write_manifest_atomic(manifest)
    print(
        f"Wrote {RAW_MANIFEST_PATH.relative_to(PROJECT_ROOT)} "
        f"for {len(manifest['files'])} files and {manifest['total_rows']:,} rows"
    )


if __name__ == "__main__":
    main()
