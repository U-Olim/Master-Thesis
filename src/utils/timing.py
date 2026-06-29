"""Lightweight runtime profiling helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypedDict

import numpy as np


GENERAL_RUNTIME_STAGES: tuple[str, ...] = (
    "runtime_total_sec",
    "runtime_data_generation_sec",
    "runtime_estimator_sec",
    "runtime_alpha_grid_sec",
    "runtime_confidence_region_sec",
    "runtime_score_eval_sec",
    "runtime_other_sec",
)

DML_RUNTIME_STAGES: tuple[str, ...] = (
    "dml_runtime_total_sec",
    "dml_runtime_crossfit_sec",
    "dml_runtime_nuisance_fit_sec",
    "dml_runtime_nuisance_predict_sec",
    "dml_runtime_alpha_loop_sec",
    "dml_runtime_score_eval_sec",
    "dml_runtime_confidence_region_sec",
)

POST_SELECTION_RUNTIME_STAGES: tuple[str, ...] = (
    "ps_runtime_total_sec",
    "ps_runtime_selection_sec",
    "ps_runtime_first_stage_sec",
    "ps_runtime_alpha_loop_sec",
    "ps_runtime_score_eval_sec",
    "ps_runtime_confidence_region_sec",
    "ps_runtime_diagnostics_sec",
)

POST_SELECTION_QUANTILE_RUNTIME_STAGES: tuple[str, ...] = (
    "psq_runtime_total_sec",
    "psq_runtime_quantile_selection_sec",
    "psq_runtime_treatment_selection_sec",
    "psq_runtime_alpha_loop_sec",
    "psq_runtime_score_eval_sec",
    "psq_runtime_confidence_region_sec",
    "psq_runtime_diagnostics_sec",
)

POST_SELECTION_ALIGNED_RUNTIME_STAGES: tuple[str, ...] = (
    "psa_runtime_total_sec",
    "psa_runtime_anchor_selection_sec",
    "psa_runtime_treatment_selection_sec",
    "psa_runtime_alpha_loop_sec",
    "psa_runtime_score_eval_sec",
    "psa_runtime_confidence_region_sec",
    "psa_runtime_diagnostics_sec",
)

ORACLE_RUNTIME_STAGES: tuple[str, ...] = (
    "oracle_runtime_total_sec",
    "oracle_runtime_alpha_loop_sec",
    "oracle_runtime_score_eval_sec",
    "oracle_runtime_confidence_region_sec",
)

FULL_CONTROL_RUNTIME_STAGES: tuple[str, ...] = (
    "fc_runtime_total_sec",
    "fc_runtime_alpha_loop_sec",
    "fc_runtime_score_eval_sec",
    "fc_runtime_confidence_region_sec",
)

RUNTIME_COLUMNS: tuple[str, ...] = (
    GENERAL_RUNTIME_STAGES
    + DML_RUNTIME_STAGES
    + POST_SELECTION_RUNTIME_STAGES
    + POST_SELECTION_QUANTILE_RUNTIME_STAGES
    + POST_SELECTION_ALIGNED_RUNTIME_STAGES
    + ORACLE_RUNTIME_STAGES
    + FULL_CONTROL_RUNTIME_STAGES
)


class RuntimeDiagnosticColumns(TypedDict, total=False):
    """Runtime diagnostic keyword fields accepted by EstimationResult."""

    runtime_total_sec: float
    runtime_data_generation_sec: float
    runtime_estimator_sec: float
    runtime_alpha_grid_sec: float
    runtime_confidence_region_sec: float
    runtime_score_eval_sec: float
    runtime_other_sec: float
    dml_runtime_total_sec: float
    dml_runtime_crossfit_sec: float
    dml_runtime_nuisance_fit_sec: float
    dml_runtime_nuisance_predict_sec: float
    dml_runtime_alpha_loop_sec: float
    dml_runtime_score_eval_sec: float
    dml_runtime_confidence_region_sec: float
    ps_runtime_total_sec: float
    ps_runtime_selection_sec: float
    ps_runtime_first_stage_sec: float
    ps_runtime_alpha_loop_sec: float
    ps_runtime_score_eval_sec: float
    ps_runtime_confidence_region_sec: float
    ps_runtime_diagnostics_sec: float
    psq_runtime_total_sec: float
    psq_runtime_quantile_selection_sec: float
    psq_runtime_treatment_selection_sec: float
    psq_runtime_alpha_loop_sec: float
    psq_runtime_score_eval_sec: float
    psq_runtime_confidence_region_sec: float
    psq_runtime_diagnostics_sec: float
    psa_runtime_total_sec: float
    psa_runtime_anchor_selection_sec: float
    psa_runtime_treatment_selection_sec: float
    psa_runtime_alpha_loop_sec: float
    psa_runtime_score_eval_sec: float
    psa_runtime_confidence_region_sec: float
    psa_runtime_diagnostics_sec: float
    oracle_runtime_total_sec: float
    oracle_runtime_alpha_loop_sec: float
    oracle_runtime_score_eval_sec: float
    oracle_runtime_confidence_region_sec: float
    fc_runtime_total_sec: float
    fc_runtime_alpha_loop_sec: float
    fc_runtime_score_eval_sec: float
    fc_runtime_confidence_region_sec: float


@dataclass
class RuntimeProfile:
    """Accumulate elapsed seconds for named runtime stages."""

    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (
                perf_counter() - start
            )

    def get(self, name: str) -> float:
        return float(self.timings.get(name, np.nan))

    def to_prefixed_dict(
        self,
        prefix: str,
        keys: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        return {f"{prefix}_{key}_sec": self.get(key) for key in keys}


def empty_runtime_columns() -> RuntimeDiagnosticColumns:
    """Return all runtime profiling columns with missing values."""
    missing = float(np.nan)
    return {
        "runtime_total_sec": missing,
        "runtime_data_generation_sec": missing,
        "runtime_estimator_sec": missing,
        "runtime_alpha_grid_sec": missing,
        "runtime_confidence_region_sec": missing,
        "runtime_score_eval_sec": missing,
        "runtime_other_sec": missing,
        "dml_runtime_total_sec": missing,
        "dml_runtime_crossfit_sec": missing,
        "dml_runtime_nuisance_fit_sec": missing,
        "dml_runtime_nuisance_predict_sec": missing,
        "dml_runtime_alpha_loop_sec": missing,
        "dml_runtime_score_eval_sec": missing,
        "dml_runtime_confidence_region_sec": missing,
        "ps_runtime_total_sec": missing,
        "ps_runtime_selection_sec": missing,
        "ps_runtime_first_stage_sec": missing,
        "ps_runtime_alpha_loop_sec": missing,
        "ps_runtime_score_eval_sec": missing,
        "ps_runtime_confidence_region_sec": missing,
        "ps_runtime_diagnostics_sec": missing,
        "psq_runtime_total_sec": missing,
        "psq_runtime_quantile_selection_sec": missing,
        "psq_runtime_treatment_selection_sec": missing,
        "psq_runtime_alpha_loop_sec": missing,
        "psq_runtime_score_eval_sec": missing,
        "psq_runtime_confidence_region_sec": missing,
        "psq_runtime_diagnostics_sec": missing,
        "psa_runtime_total_sec": missing,
        "psa_runtime_anchor_selection_sec": missing,
        "psa_runtime_treatment_selection_sec": missing,
        "psa_runtime_alpha_loop_sec": missing,
        "psa_runtime_score_eval_sec": missing,
        "psa_runtime_confidence_region_sec": missing,
        "psa_runtime_diagnostics_sec": missing,
        "oracle_runtime_total_sec": missing,
        "oracle_runtime_alpha_loop_sec": missing,
        "oracle_runtime_score_eval_sec": missing,
        "oracle_runtime_confidence_region_sec": missing,
        "fc_runtime_total_sec": missing,
        "fc_runtime_alpha_loop_sec": missing,
        "fc_runtime_score_eval_sec": missing,
        "fc_runtime_confidence_region_sec": missing,
    }


def _nonnegative_or_nan(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        return float(np.nan)
    return value


def estimator_runtime_columns(
    *,
    estimator: str,
    total_sec: float,
    alpha_loop_sec: float = np.nan,
    confidence_region_sec: float = np.nan,
    score_eval_sec: float = np.nan,
    data_generation_sec: float = np.nan,
    crossfit_sec: float = np.nan,
    nuisance_fit_sec: float = np.nan,
    nuisance_predict_sec: float = np.nan,
    selection_sec: float = np.nan,
    first_stage_sec: float = np.nan,
    diagnostics_sec: float = np.nan,
    quantile_selection_sec: float = np.nan,
    treatment_selection_sec: float = np.nan,
    anchor_selection_sec: float = np.nan,
) -> RuntimeDiagnosticColumns:
    """Build complete runtime columns for one estimator result."""
    total_sec = _nonnegative_or_nan(total_sec)
    alpha_loop_sec = _nonnegative_or_nan(alpha_loop_sec)
    confidence_region_sec = _nonnegative_or_nan(confidence_region_sec)
    score_eval_sec = _nonnegative_or_nan(score_eval_sec)
    data_generation_sec = _nonnegative_or_nan(data_generation_sec)
    crossfit_sec = _nonnegative_or_nan(crossfit_sec)
    nuisance_fit_sec = _nonnegative_or_nan(nuisance_fit_sec)
    nuisance_predict_sec = _nonnegative_or_nan(nuisance_predict_sec)
    selection_sec = _nonnegative_or_nan(selection_sec)
    first_stage_sec = _nonnegative_or_nan(first_stage_sec)
    diagnostics_sec = _nonnegative_or_nan(diagnostics_sec)
    quantile_selection_sec = _nonnegative_or_nan(quantile_selection_sec)
    treatment_selection_sec = _nonnegative_or_nan(treatment_selection_sec)
    anchor_selection_sec = _nonnegative_or_nan(anchor_selection_sec)

    columns = empty_runtime_columns()
    columns["runtime_total_sec"] = total_sec
    columns["runtime_estimator_sec"] = total_sec
    columns["runtime_data_generation_sec"] = data_generation_sec
    columns["runtime_alpha_grid_sec"] = alpha_loop_sec
    columns["runtime_confidence_region_sec"] = confidence_region_sec
    columns["runtime_score_eval_sec"] = score_eval_sec

    known = [
        alpha_loop_sec,
        confidence_region_sec,
        score_eval_sec,
        data_generation_sec,
        selection_sec,
        first_stage_sec,
        diagnostics_sec,
        quantile_selection_sec,
        treatment_selection_sec,
        anchor_selection_sec,
        crossfit_sec,
    ]
    finite_known = [value for value in known if np.isfinite(value)]
    if np.isfinite(total_sec) and finite_known:
        other = total_sec - sum(finite_known)
        columns["runtime_other_sec"] = float(other) if other >= 0 else float(np.nan)

    if estimator == "dml_ivqr":
        columns.update(
            {
                "dml_runtime_total_sec": total_sec,
                "dml_runtime_crossfit_sec": crossfit_sec,
                "dml_runtime_nuisance_fit_sec": nuisance_fit_sec,
                "dml_runtime_nuisance_predict_sec": nuisance_predict_sec,
                "dml_runtime_alpha_loop_sec": alpha_loop_sec,
                "dml_runtime_score_eval_sec": score_eval_sec,
                "dml_runtime_confidence_region_sec": confidence_region_sec,
            }
        )
    elif estimator == "post_selection_ivqr":
        columns.update(
            {
                "ps_runtime_total_sec": total_sec,
                "ps_runtime_selection_sec": selection_sec,
                "ps_runtime_first_stage_sec": first_stage_sec,
                "ps_runtime_alpha_loop_sec": alpha_loop_sec,
                "ps_runtime_score_eval_sec": score_eval_sec,
                "ps_runtime_confidence_region_sec": confidence_region_sec,
                "ps_runtime_diagnostics_sec": diagnostics_sec,
            }
        )
    elif estimator == "post_selection_quantile":
        columns.update(
            {
                "psq_runtime_total_sec": total_sec,
                "psq_runtime_quantile_selection_sec": quantile_selection_sec,
                "psq_runtime_treatment_selection_sec": treatment_selection_sec,
                "psq_runtime_alpha_loop_sec": alpha_loop_sec,
                "psq_runtime_score_eval_sec": score_eval_sec,
                "psq_runtime_confidence_region_sec": confidence_region_sec,
                "psq_runtime_diagnostics_sec": diagnostics_sec,
            }
        )
    elif estimator == "post_selection_ivqr_aligned":
        columns.update(
            {
                "psa_runtime_total_sec": total_sec,
                "psa_runtime_anchor_selection_sec": anchor_selection_sec,
                "psa_runtime_treatment_selection_sec": treatment_selection_sec,
                "psa_runtime_alpha_loop_sec": alpha_loop_sec,
                "psa_runtime_score_eval_sec": score_eval_sec,
                "psa_runtime_confidence_region_sec": confidence_region_sec,
                "psa_runtime_diagnostics_sec": diagnostics_sec,
            }
        )
    elif estimator == "oracle":
        columns.update(
            {
                "oracle_runtime_total_sec": total_sec,
                "oracle_runtime_alpha_loop_sec": alpha_loop_sec,
                "oracle_runtime_score_eval_sec": score_eval_sec,
                "oracle_runtime_confidence_region_sec": confidence_region_sec,
            }
        )
    elif estimator == "full_control_ivqr":
        columns.update(
            {
                "fc_runtime_total_sec": total_sec,
                "fc_runtime_alpha_loop_sec": alpha_loop_sec,
                "fc_runtime_score_eval_sec": score_eval_sec,
                "fc_runtime_confidence_region_sec": confidence_region_sec,
            }
        )
    return columns


__all__ = [
    "DML_RUNTIME_STAGES",
    "FULL_CONTROL_RUNTIME_STAGES",
    "GENERAL_RUNTIME_STAGES",
    "ORACLE_RUNTIME_STAGES",
    "POST_SELECTION_RUNTIME_STAGES",
    "POST_SELECTION_QUANTILE_RUNTIME_STAGES",
    "POST_SELECTION_ALIGNED_RUNTIME_STAGES",
    "RUNTIME_COLUMNS",
    "RuntimeDiagnosticColumns",
    "RuntimeProfile",
    "empty_runtime_columns",
    "estimator_runtime_columns",
]
