"""Load, validate, and harmonize the completed R=500 result files."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_RESULT_FILES = {
    "oracle": PROJECT_ROOT / "results" / "raw" / "oracle_ivqr" / "oracle_ivqr.csv",
    "post_selection": (
        PROJECT_ROOT
        / "results"
        / "raw"
        / "post_selection_ivqr"
        / "post_selection_ivqr.csv"
    ),
    "dml": PROJECT_ROOT / "results" / "raw" / "dml_ivqr" / "dml_ivqr.csv",
}
RAW_MANIFEST_PATH = PROJECT_ROOT / "results" / "raw" / "manifest.json"
RAW_ESTIMATOR_LABELS = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}

IDENTIFIER_COLUMNS = ["estimator", "dgp", "n", "p", "pi", "tau", "rep"]
DESIGN_COLUMNS = ["estimator", "dgp", "n", "p", "pi", "tau"]
CORE_COLUMNS = [
    *IDENTIFIER_COLUMNS,
    "seed",
    "alpha_hat",
    "alpha_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "covered",
    "converged",
]
SELECTION_COLUMNS = ["n_selected_controls", "selection_lasso_multiplier"]
COMMON_COLUMNS = [*CORE_COLUMNS, *SELECTION_COLUMNS]
NUMERIC_COLUMNS = [
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "alpha_hat",
    "alpha_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
]


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest using bounded-memory reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def verify_raw_manifest(
    manifest_path: str | Path = RAW_MANIFEST_PATH,
    raw_result_files: Mapping[str, Path] | None = None,
) -> None:
    """Verify canonical raw files against their tracked provenance manifest."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Raw-result manifest does not exist: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read raw-result manifest {manifest}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError(f"Raw-result manifest has an invalid structure: {manifest}")

    expected_files = RAW_RESULT_FILES if raw_result_files is None else raw_result_files
    entries: dict[str, dict[str, object]] = {}
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("estimator"), str):
            raise ValueError(f"Raw-result manifest has an invalid file entry: {item!r}")
        estimator = item["estimator"]
        if estimator in entries:
            raise ValueError(f"Raw-result manifest repeats estimator: {estimator}")
        entries[estimator] = item

    expected_estimators = set(expected_files)
    if set(entries) != expected_estimators:
        raise ValueError(
            "Raw-result manifest estimator mismatch: "
            f"expected {sorted(expected_estimators)}, observed {sorted(entries)}"
        )
    if payload.get("number_of_files") != len(expected_files):
        raise ValueError(
            "Raw-result manifest number_of_files mismatch: "
            f"expected {len(expected_files)}, observed {payload.get('number_of_files')}"
        )

    for estimator, source in expected_files.items():
        entry = entries[estimator]
        expected_path = _manifest_file_path(source)
        if entry.get("path") != expected_path:
            raise ValueError(
                f"Raw-result path mismatch for {estimator}: expected {expected_path}, "
                f"observed {entry.get('path')}"
            )
        if entry.get("filename") != source.name:
            raise ValueError(
                f"Raw-result filename mismatch for {expected_path}: expected {source.name}, "
                f"observed {entry.get('filename')}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"Canonical raw result does not exist: {source}")
        observed_size = source.stat().st_size
        if entry.get("size_bytes") != observed_size:
            raise ValueError(
                f"Raw-result size mismatch for {expected_path}: "
                f"expected {entry.get('size_bytes')}, observed {observed_size}"
            )
        observed_hash = sha256_file(source)
        if entry.get("sha256") != observed_hash:
            raise ValueError(
                f"Raw-result SHA-256 mismatch for {expected_path}: "
                f"expected {entry.get('sha256')}, observed {observed_hash}"
            )


def require_unique_selection_lasso_multiplier(values: pd.Series) -> float:
    """Return the sole nonmissing final-experiment Lasso multiplier."""
    unique_values = values.dropna().unique()
    if len(unique_values) != 1:
        raise ValueError(
            "Expected exactly one unique selection_lasso_multiplier "
            f"in final post-selection results, found {len(unique_values)}: "
            f"{sorted(unique_values.tolist())}"
        )
    return float(unique_values[0])


def _read_results(
    path: str | Path,
    estimator: str,
    *,
    expected_replications: int,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Result file does not exist: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Result file is empty: {source}")

    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Result file contains no rows: {source}")

    missing = set(CORE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    raw_labels = set(frame["estimator"].dropna().unique())
    expected_label = RAW_ESTIMATOR_LABELS[estimator]
    if raw_labels != {expected_label}:
        raise ValueError(
            f"{source} has estimator labels {sorted(raw_labels)!r}; "
            f"expected only {expected_label!r}"
        )

    frame = frame.copy()
    frame["estimator"] = estimator
    for column in SELECTION_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.loc[:, COMMON_COLUMNS]
    validate_results(
        frame,
        expected_estimator=estimator,
        expected_replications=expected_replications,
    )
    return frame


def validate_results(
    results: pd.DataFrame,
    *,
    expected_estimator: str | None = None,
    expected_replications: int = 500,
) -> None:
    """Validate harmonized results, raising ``ValueError`` on any violation.

    Confidence regions are grid-inverted and may be disconnected. The reported
    bounds are therefore the hull, while ``cr_length`` is total accepted-set
    length; equality between hull width and length is not required.
    """
    if results.empty:
        raise ValueError("Results contain no rows")
    missing = set(CORE_COLUMNS).difference(results.columns)
    if missing:
        raise ValueError(f"Results are missing required columns: {sorted(missing)}")
    if expected_replications < 1:
        raise ValueError("expected_replications must be positive")

    labels = set(results["estimator"].dropna().unique())
    allowed = set(RAW_RESULT_FILES)
    if not labels or not labels.issubset(allowed):
        raise ValueError(f"Unexpected estimator labels: {sorted(labels)!r}")
    if expected_estimator is not None and labels != {expected_estimator}:
        raise ValueError(
            f"Expected estimator {expected_estimator!r}, observed {sorted(labels)!r}"
        )

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(results[column]):
            raise ValueError(f"{column} must be numeric")
    for column in ("covered", "converged"):
        if not pd.api.types.is_bool_dtype(results[column]):
            raise ValueError(f"{column} must be boolean")

    required_complete = [
        "estimator",
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "rep",
        "seed",
        "alpha_hat",
        "alpha_true",
        "covered",
        "converged",
    ]
    if results[required_complete].isna().any().any():
        columns = results[required_complete].columns[
            results[required_complete].isna().any()
        ].tolist()
        raise ValueError(f"Required values are missing in columns: {columns}")
    finite_columns = ["n", "p", "pi", "tau", "rep", "seed", "alpha_hat", "alpha_true"]
    if not np.isfinite(results[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("Core numeric values must be finite")

    for column in ("n", "p", "rep", "seed"):
        values = results[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integers")
    if (results["n"] <= 0).any() or (results["p"] < 0).any():
        raise ValueError("n must be positive and p must be nonnegative")
    if (results["pi"] <= 0).any():
        raise ValueError("pi must be positive")
    if ((results["tau"] <= 0) | (results["tau"] >= 1)).any():
        raise ValueError("tau must lie strictly between zero and one")

    duplicates = results.duplicated(IDENTIFIER_COLUMNS)
    if duplicates.any():
        raise ValueError(
            f"Found {int(duplicates.sum())} duplicate Monte Carlo identifiers"
        )

    rep_summary = results.groupby(DESIGN_COLUMNS, dropna=False)["rep"].agg(
        ["size", "nunique", "min", "max"]
    )
    expected_max = expected_replications - 1
    bad_replications = rep_summary[
        (rep_summary["size"] != expected_replications)
        | (rep_summary["nunique"] != expected_replications)
        | (rep_summary["min"] != 0)
        | (rep_summary["max"] != expected_max)
    ]
    if not bad_replications.empty:
        example = bad_replications.head(5).reset_index().to_dict("records")
        raise ValueError(
            f"{len(bad_replications)} design cells do not contain replications "
            f"0-{expected_max}; examples: {example}"
        )

    cr = results[["cr_lower", "cr_upper", "cr_length"]]
    missing_cr = cr.isna()
    partial_cr = missing_cr.any(axis=1) & ~missing_cr.all(axis=1)
    if partial_cr.any():
        raise ValueError(f"Found {int(partial_cr.sum())} partially missing confidence regions")
    complete_cr = ~missing_cr.any(axis=1)
    finite_cr = np.isfinite(cr.loc[complete_cr].to_numpy(dtype=float)).all(axis=1)
    if not finite_cr.all():
        raise ValueError("Nonmissing confidence-region values must be finite")

    lower = results.loc[complete_cr, "cr_lower"]
    upper = results.loc[complete_cr, "cr_upper"]
    length = results.loc[complete_cr, "cr_length"]
    hull_length = upper - lower
    if (lower > upper).any():
        raise ValueError("cr_lower must not exceed cr_upper")
    if (length < -1e-10).any() or (length > hull_length + 1e-9).any():
        raise ValueError("cr_length must be nonnegative and no greater than its hull")
    impossible_coverage = results.loc[complete_cr, "covered"] & ~results.loc[
        complete_cr, "alpha_true"
    ].between(lower, upper, inclusive="both")
    if impossible_coverage.any():
        raise ValueError("covered=True is inconsistent with confidence-region bounds")
    if results.loc[~complete_cr, "covered"].any():
        raise ValueError("A missing confidence region cannot have covered=True")

    true_spread = results.groupby(["dgp", "tau"], dropna=False)["alpha_true"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (true_spread > 1e-12).any():
        raise ValueError("alpha_true is not constant within DGP/quantile cells")

    if "post_selection" in labels:
        for column in SELECTION_COLUMNS:
            if column not in results:
                raise ValueError(f"Post-selection results require {column}")
        selected_rows = results["estimator"].eq("post_selection")
        selected = results.loc[selected_rows, "n_selected_controls"]
        multiplier = results.loc[selected_rows, "selection_lasso_multiplier"]
        require_unique_selection_lasso_multiplier(multiplier)
        if selected.isna().any() or multiplier.isna().any():
            raise ValueError("Post-selection diagnostics must not be missing")
        if ((selected < 0) | (selected > results.loc[selected_rows, "p"])).any():
            raise ValueError("n_selected_controls must lie between zero and p")
        if not np.equal(selected, np.floor(selected)).all():
            raise ValueError("n_selected_controls must contain integers")
        if (~np.isfinite(multiplier) | (multiplier <= 0)).any():
            raise ValueError("selection_lasso_multiplier must be finite and positive")


def load_oracle_results(
    path: str | Path = RAW_RESULT_FILES["oracle"],
    *,
    expected_replications: int = 500,
) -> pd.DataFrame:
    return _read_results(path, "oracle", expected_replications=expected_replications)


def load_post_selection_results(
    path: str | Path = RAW_RESULT_FILES["post_selection"],
    *,
    expected_replications: int = 500,
) -> pd.DataFrame:
    return _read_results(
        path, "post_selection", expected_replications=expected_replications
    )


def load_dml_results(
    path: str | Path = RAW_RESULT_FILES["dml"],
    *,
    expected_replications: int = 500,
) -> pd.DataFrame:
    return _read_results(path, "dml", expected_replications=expected_replications)


def load_all_results(*, expected_replications: int = 500) -> pd.DataFrame:
    """Load and validate all three completed result datasets."""
    results = pd.concat(
        [
            load_oracle_results(expected_replications=expected_replications),
            load_post_selection_results(expected_replications=expected_replications),
            load_dml_results(expected_replications=expected_replications),
        ],
        ignore_index=True,
    )
    validate_results(results, expected_replications=expected_replications)
    return results
