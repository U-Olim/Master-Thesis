"""Load, validate, and harmonize the completed R=500 result files."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from simulation.dml_output import validate_component_columns
from simulation.oracle_output import (
    ORACLE_OUTPUT_COLUMNS,
    clean_oracle_results_frame,
)


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
CR_REPORTING_COLUMNS = [
    "result_schema_version",
    "cr_components",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_status",
    "cr_is_numerically_resolved",
    "cr_unresolved_count",
    "cr_unresolved_alphas",
]
COMMON_COLUMNS = [*CORE_COLUMNS, *CR_REPORTING_COLUMNS, *SELECTION_COLUMNS]
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


def sha256_canonical_lf_file(
    path: str | Path, *, chunk_size: int = 1024 * 1024
) -> str:
    """Hash bytes after normalizing CRLF and standalone CR line endings to LF."""
    digest = hashlib.sha256()
    pending_cr = False
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            if pending_cr:
                digest.update(b"\n")
                if chunk.startswith(b"\n"):
                    chunk = chunk[1:]
                pending_cr = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_cr = True
            digest.update(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    if pending_cr:
        digest.update(b"\n")
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
    supported_schema_version = 3
    if payload.get("schema_version") != supported_schema_version:
        raise ValueError(
            "Raw-result manifest schema mismatch: "
            f"observed {payload.get('schema_version')}, supported version is "
            f"{supported_schema_version}"
        )
    final_run_metadata = payload.get("final_run_metadata")
    if not isinstance(final_run_metadata, Mapping):
        raise ValueError("Raw-result manifest final_run_metadata must be an object")
    if not isinstance(final_run_metadata.get("common"), Mapping):
        raise ValueError("Raw-result manifest final_run_metadata.common must be an object")
    estimator_metadata = final_run_metadata.get("estimators")
    if not isinstance(estimator_metadata, Mapping):
        raise ValueError(
            "Raw-result manifest final_run_metadata.estimators must be an object"
        )
    canonical_estimators = set(RAW_RESULT_FILES)
    metadata_estimators = set(estimator_metadata)
    missing_metadata = canonical_estimators - metadata_estimators
    if missing_metadata:
        raise ValueError(
            "Raw-result manifest final_run_metadata.estimators is missing keys: "
            f"{sorted(missing_metadata)}"
        )
    unexpected_metadata = metadata_estimators - canonical_estimators
    if unexpected_metadata:
        raise ValueError(
            "Raw-result manifest final_run_metadata.estimators has unexpected keys: "
            f"{sorted(unexpected_metadata)}"
        )
    for estimator, settings in estimator_metadata.items():
        if not isinstance(settings, Mapping):
            raise ValueError(
                "Raw-result manifest estimator metadata must be an object: "
                f"{estimator}"
            )

    expected_files = RAW_RESULT_FILES if raw_result_files is None else raw_result_files
    manifest_file_count = payload.get("number_of_files")
    if manifest_file_count != len(payload["files"]):
        raise ValueError(
            "Raw-result manifest number_of_files mismatch with files list: "
            f"recorded {manifest_file_count}, observed {len(payload['files'])} entries"
        )
    entries: dict[str, dict[str, object]] = {}
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("estimator"), str):
            raise ValueError(f"Raw-result manifest has an invalid file entry: {item!r}")
        estimator = item["estimator"]
        if estimator in entries:
            raise ValueError(f"Raw-result manifest repeats estimator: {estimator}")
        entries[estimator] = item

    expected_estimators = set(expected_files)
    observed_estimators = set(entries)
    missing_estimators = expected_estimators - observed_estimators
    if missing_estimators:
        raise ValueError(
            "Raw-result manifest is missing estimator entries: "
            f"{sorted(missing_estimators)}"
        )
    unexpected_estimators = observed_estimators - expected_estimators
    if unexpected_estimators:
        raise ValueError(
            "Raw-result manifest has unexpected estimator entries: "
            f"{sorted(unexpected_estimators)}"
        )
    if manifest_file_count != len(expected_files):
        raise ValueError(
            "Raw-result manifest number_of_files mismatch with canonical estimators: "
            f"recorded {manifest_file_count}, expected {len(expected_files)}"
        )

    manifest_rows: list[int] = []
    for estimator, entry in entries.items():
        rows = entry.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise ValueError(
                f"Raw-result manifest rows for {estimator} must be a nonnegative integer"
            )
        manifest_rows.append(rows)
    manifest_total_rows = payload.get("total_rows")
    if not isinstance(manifest_total_rows, int) or isinstance(
        manifest_total_rows, bool
    ):
        raise ValueError("Raw-result manifest total_rows must be an integer")
    summed_manifest_rows = sum(manifest_rows)
    if manifest_total_rows != summed_manifest_rows:
        raise ValueError(
            "Raw-result manifest total_rows mismatch with file entries: "
            f"recorded {manifest_total_rows}, summed rows {summed_manifest_rows}"
        )

    actual_total_rows = 0
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
        frame = pd.read_csv(source)
        observed_rows = len(frame)
        observed_columns = len(frame.columns)
        expected_rows = entry.get("rows")
        expected_columns = entry.get("columns")
        if expected_rows != observed_rows:
            raise ValueError(
                f"Raw-result row-count mismatch for {expected_path}: "
                f"expected {expected_rows}, observed {observed_rows}"
            )
        if expected_columns != observed_columns:
            raise ValueError(
                f"Raw-result column-count mismatch for {expected_path}: "
                f"expected {expected_columns}, observed {observed_columns}"
            )
        actual_total_rows += observed_rows

        if estimator == "post_selection" and "selection_lasso_multiplier" in frame:
            documented_multiplier = estimator_metadata["post_selection"].get(
                "selection_lasso_multiplier"
            )
            if (
                not isinstance(documented_multiplier, (int, float))
                or isinstance(documented_multiplier, bool)
                or not np.isfinite(documented_multiplier)
            ):
                raise ValueError(
                    "Post-selection provenance must contain a finite numeric "
                    "selection_lasso_multiplier"
                )
            observed_multiplier = require_unique_selection_lasso_multiplier(
                frame["selection_lasso_multiplier"]
            )
            if not np.isclose(
                observed_multiplier,
                float(documented_multiplier),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Post-selection selection_lasso_multiplier conflicts with "
                    f"provenance metadata: raw data has {observed_multiplier}, "
                    f"metadata has {documented_multiplier}"
                )

        observed_size = source.stat().st_size
        observed_byte_hash = sha256_file(source)
        exact_match = (
            entry.get("size_bytes") == observed_size
            and entry.get("sha256_bytes") == observed_byte_hash
        )
        if exact_match:
            continue

        observed_canonical_hash = sha256_canonical_lf_file(source)
        if entry.get("sha256_canonical_lf") != observed_canonical_hash:
            raise ValueError(
                f"Raw-result canonical LF SHA-256 mismatch for {expected_path}: "
                f"expected {entry.get('sha256_canonical_lf')}, "
                f"observed {observed_canonical_hash}; exact-byte metadata was "
                f"size {entry.get('size_bytes')} / {entry.get('sha256_bytes')}, "
                f"observed {observed_size} / {observed_byte_hash}"
            )
        warnings.warn(
            f"Raw-result exact bytes differ for {expected_path}, but canonical LF "
            "content matches; accepting the newline-equivalent representation",
            UserWarning,
            stacklevel=2,
        )

    if manifest_total_rows != actual_total_rows:
        raise ValueError(
            "Raw-result manifest total_rows mismatch with canonical files: "
            f"expected {manifest_total_rows}, observed {actual_total_rows}"
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

    if estimator == "oracle" and set(ORACLE_OUTPUT_COLUMNS).issubset(frame.columns):
        frame = clean_oracle_results_frame(frame)
        frame.insert(0, "estimator", "oracle")
        resolved = frame["cr_is_numerically_resolved"].eq(True) & frame[
            "covered"
        ].notna()
        frame["coverage_status"] = np.where(
            resolved,
            np.where(frame["covered"].eq(True), "covered", "not_covered"),
            "coverage_unresolved",
        )
        validate_results(
            frame,
            expected_estimator="oracle",
            expected_replications=expected_replications,
        )
        return frame

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
    cr_defaults: dict[str, object] = {
        "result_schema_version": np.nan,
        "cr_components": None,
        "cr_n_blocks": np.nan,
        "cr_disconnected": np.nan,
        "cr_status": "unavailable",
        "cr_is_numerically_resolved": np.nan,
        "cr_unresolved_count": np.nan,
        "cr_unresolved_alphas": None,
    }
    for column in CR_REPORTING_COLUMNS:
        if column not in frame:
            frame[column] = cr_defaults[column]
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
    if expected_replications < 1:
        raise ValueError("expected_replications must be positive")
    current_oracle = set(ORACLE_OUTPUT_COLUMNS).issubset(results.columns) and not set(
        CORE_COLUMNS
    ).issubset(results.columns)
    if current_oracle:
        oracle = clean_oracle_results_frame(results.loc[:, ORACLE_OUTPUT_COLUMNS])
        if "estimator" in results and results["estimator"].ne("oracle").any():
            raise ValueError("Oracle results contain a non-Oracle estimator")
        design_columns = ["dgp", "n", "p", "pi", "tau"]
        rep_summary = oracle.groupby(design_columns, dropna=False)["rep"].agg(
            ["size", "nunique", "min", "max"]
        )
        expected_max = expected_replications - 1
        bad = rep_summary[
            (rep_summary["size"] != expected_replications)
            | (rep_summary["nunique"] != expected_replications)
            | (rep_summary["min"] != 0)
            | (rep_summary["max"] != expected_max)
        ]
        if not bad.empty:
            raise ValueError(
                f"{len(bad)} Oracle design cells do not contain replications "
                f"0-{expected_max}"
            )
        return
    missing = set(CORE_COLUMNS).difference(results.columns)
    if missing:
        raise ValueError(f"Results are missing required columns: {sorted(missing)}")

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
        "alpha_true",
        "covered",
        "converged",
    ]
    if results[required_complete].isna().any().any():
        columns = results[required_complete].columns[
            results[required_complete].isna().any()
        ].tolist()
        raise ValueError(f"Required values are missing in columns: {columns}")
    finite_columns = ["n", "p", "pi", "tau", "rep", "seed", "alpha_true"]
    if not np.isfinite(results[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("Core numeric values must be finite")

    successful = results["converged"]
    successful_estimates = results.loc[successful, "alpha_hat"]
    if successful_estimates.isna().any() or (
        ~np.isfinite(successful_estimates.to_numpy(dtype=float))
    ).any():
        raise ValueError("Successful rows must have a finite alpha_hat")

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
    if results.loc[~successful, "covered"].any():
        raise ValueError("Failed rows cannot have covered=True")
    if set(CR_REPORTING_COLUMNS).issubset(results.columns):
        validate_component_columns(results)

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
        successful_selected_rows = selected_rows & successful
        all_selected = results.loc[selected_rows, "n_selected_controls"]
        all_multipliers = results.loc[
            selected_rows, "selection_lasso_multiplier"
        ]
        selected = results.loc[successful_selected_rows, "n_selected_controls"]
        multiplier = results.loc[
            successful_selected_rows, "selection_lasso_multiplier"
        ]
        require_unique_selection_lasso_multiplier(all_multipliers)
        if selected.isna().any() or multiplier.isna().any():
            raise ValueError(
                "Successful post-selection diagnostics must not be missing"
            )
        present_selected = all_selected.notna()
        if (
            (all_selected.loc[present_selected] < 0)
            | (
                all_selected.loc[present_selected]
                > results.loc[all_selected.index[present_selected], "p"]
            )
        ).any():
            raise ValueError("n_selected_controls must lie between zero and p")
        if not np.equal(
            all_selected.loc[present_selected],
            np.floor(all_selected.loc[present_selected]),
        ).all():
            raise ValueError("n_selected_controls must contain integers")
        present_multiplier = all_multipliers.notna()
        if (
            ~np.isfinite(all_multipliers.loc[present_multiplier])
            | (all_multipliers.loc[present_multiplier] <= 0)
        ).any():
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
