"""Defaults and typed configuration for the IVQR Monte Carlo study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

N_VALUES: tuple[int, ...] = (500, 1000)
P_VALUES: tuple[int, ...] = (200, 500)
PI_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)
DGPS: tuple[str, ...] = ("dgp1", "dgp2", "dgp3")

ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")
DEFAULT_ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")

R_FAST: int = 10

DEFAULT_ALPHA_MIN: float = -1.0
DEFAULT_ALPHA_MAX: float = 3.0
DEFAULT_ALPHA_GRID_SIZE: int = 21

DEFAULT_N_JOBS: int = 4
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_BASE_SEED: int = 12345
DEFAULT_QUANTREG_MAX_ITER: int = 1000
DEFAULT_DML_K_FOLDS: int = 3
DEFAULT_DML_QUANTILE_PENALTY: float = 0.01
DEFAULT_DML_RIDGE_ALPHA: float = 1.0
DEFAULT_DML_QUANTILE_SOLVER: str = "highs"
DEFAULT_CRITICAL_VALUE_MULTIPLIER: float = 1.0
DEFAULT_SELECTION_LASSO_MULTIPLIER: float = 1.0
# Shared CH production defaults. Fixed/reject/legacy_reject remain explicit
# reproducibility choices for historical simulation output.
DEFAULT_ITERATION_WARNING_POLICY: str = "use_if_valid"
DEFAULT_HARD_FAILURE_POLICY: str = "unresolved"
DEFAULT_GRID_STRATEGY: str = "adaptive"
DEFAULT_REFINEMENT_TOLERANCE: float = 0.025
DEFAULT_MAX_REFINEMENT_DEPTH: int = 10
DEFAULT_MAX_ALPHA_EVALUATIONS: int = 201
DEFAULT_ADAPTIVE_MIDPOINT_PROBE: bool = True
DEFAULT_ALPHA_HAT_GRID: str = "initial"

RHO_X: float = 0.5
RHO_UV: float = 0.5
DF_T: int = 5

FAST_OUTPUT: str = "results/raw/fast_results.csv"


@dataclass(frozen=True)
class ExecutionConfig:
    reps: int
    rep_start: int
    rep_end: int
    base_seed: int
    n_jobs: int
    batch_size: int
    output_path: Path
    manifest_path: Path | None
    max_designs: int | None
    resume: bool
    rerun_failed: bool
    dry_run: bool


@dataclass(frozen=True)
class DesignConfig:
    dgps: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    dimensions: tuple[int, ...]
    instrument_strengths: tuple[float, ...]
    quantiles: tuple[float, ...]


@dataclass(frozen=True)
class AlphaGridConfig:
    alpha_min: float
    alpha_max: float
    alpha_grid_size: int


@dataclass(frozen=True)
class CHInferenceConfig:
    iteration_warning_policy: str
    hard_failure_policy: str
    grid_strategy: str
    adaptive_midpoint_probe: bool
    refinement_tolerance: float
    max_refinement_depth: int
    max_alpha_evaluations: int
    alpha_hat_grid: str
    critical_value_multiplier: float
    quantreg_max_iter: int
    show_quantreg_warnings: bool


@dataclass(frozen=True)
class PostSelectionConfig:
    selection_lasso_multiplier: float


@dataclass(frozen=True)
class DMLConfig:
    k_folds: int
    quantile_penalty: float
    ridge_alpha: float
    quantile_solver: str
    critical_value_multiplier: float


@dataclass(frozen=True)
class OracleRunConfig:
    execution: ExecutionConfig
    design: DesignConfig
    alpha_grid: AlphaGridConfig
    inference: CHInferenceConfig


@dataclass(frozen=True)
class PostSelectionRunConfig:
    execution: ExecutionConfig
    design: DesignConfig
    alpha_grid: AlphaGridConfig
    inference: CHInferenceConfig
    selection: PostSelectionConfig


@dataclass(frozen=True)
class DMLRunConfig:
    execution: ExecutionConfig
    design: DesignConfig
    alpha_grid: AlphaGridConfig
    dml: DMLConfig


EstimatorRunConfig: TypeAlias = OracleRunConfig | PostSelectionRunConfig | DMLRunConfig


def _namespace_value(namespace: Any, name: str, default: Any) -> Any:
    value = getattr(namespace, name, None)
    return default if value is None else value


def _build_execution_config(namespace: Any) -> ExecutionConfig:
    reps = int(_namespace_value(namespace, "reps", R_FAST))
    rep_end = int(_namespace_value(namespace, "rep_end", reps - 1))
    output = _namespace_value(namespace, "output", FAST_OUTPUT)
    manifest = getattr(namespace, "manifest", None)
    return ExecutionConfig(
        reps=reps,
        rep_start=int(_namespace_value(namespace, "rep_start", 0)),
        rep_end=rep_end,
        base_seed=int(_namespace_value(namespace, "base_seed", DEFAULT_BASE_SEED)),
        n_jobs=int(_namespace_value(namespace, "n_jobs", DEFAULT_N_JOBS)),
        batch_size=int(_namespace_value(namespace, "batch_size", DEFAULT_BATCH_SIZE)),
        output_path=Path(output),
        manifest_path=None if manifest is None else Path(manifest),
        max_designs=getattr(namespace, "max_designs", None),
        resume=bool(getattr(namespace, "resume", False)),
        rerun_failed=bool(getattr(namespace, "rerun_failed", False)),
        dry_run=bool(getattr(namespace, "dry_run", False)),
    )


def _build_design_config(namespace: Any) -> DesignConfig:
    return DesignConfig(
        dgps=tuple(_namespace_value(namespace, "dgps", DGPS)),
        sample_sizes=tuple(_namespace_value(namespace, "n_values", N_VALUES)),
        dimensions=tuple(_namespace_value(namespace, "p_values", P_VALUES)),
        instrument_strengths=tuple(
            _namespace_value(namespace, "pi_values", PI_VALUES)
        ),
        quantiles=tuple(_namespace_value(namespace, "taus", TAUS)),
    )


def _build_alpha_grid_config(namespace: Any) -> AlphaGridConfig:
    return AlphaGridConfig(
        alpha_min=float(
            _namespace_value(namespace, "alpha_min", DEFAULT_ALPHA_MIN)
        ),
        alpha_max=float(
            _namespace_value(namespace, "alpha_max", DEFAULT_ALPHA_MAX)
        ),
        alpha_grid_size=int(
            _namespace_value(
                namespace, "alpha_grid_size", DEFAULT_ALPHA_GRID_SIZE
            )
        ),
    )


def _build_ch_inference_config(namespace: Any) -> CHInferenceConfig:
    return CHInferenceConfig(
        iteration_warning_policy=str(
            _namespace_value(
                namespace,
                "iteration_warning_policy",
                DEFAULT_ITERATION_WARNING_POLICY,
            )
        ),
        hard_failure_policy=str(
            _namespace_value(
                namespace, "hard_failure_policy", DEFAULT_HARD_FAILURE_POLICY
            )
        ),
        grid_strategy=str(
            _namespace_value(namespace, "grid_strategy", DEFAULT_GRID_STRATEGY)
        ),
        adaptive_midpoint_probe=bool(
            _namespace_value(
                namespace,
                "adaptive_midpoint_probe",
                DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
            )
        ),
        refinement_tolerance=float(
            _namespace_value(
                namespace,
                "refinement_tolerance",
                DEFAULT_REFINEMENT_TOLERANCE,
            )
        ),
        max_refinement_depth=int(
            _namespace_value(
                namespace,
                "max_refinement_depth",
                DEFAULT_MAX_REFINEMENT_DEPTH,
            )
        ),
        max_alpha_evaluations=int(
            _namespace_value(
                namespace,
                "max_alpha_evaluations",
                DEFAULT_MAX_ALPHA_EVALUATIONS,
            )
        ),
        alpha_hat_grid=str(
            _namespace_value(namespace, "alpha_hat_grid", DEFAULT_ALPHA_HAT_GRID)
        ),
        critical_value_multiplier=float(
            _namespace_value(
                namespace,
                "critical_value_multiplier",
                DEFAULT_CRITICAL_VALUE_MULTIPLIER,
            )
        ),
        quantreg_max_iter=int(
            _namespace_value(
                namespace, "quantreg_max_iter", DEFAULT_QUANTREG_MAX_ITER
            )
        ),
        show_quantreg_warnings=bool(
            getattr(namespace, "show_quantreg_warnings", False)
        ),
    )


def build_oracle_run_config(namespace: Any) -> OracleRunConfig:
    return OracleRunConfig(
        execution=_build_execution_config(namespace),
        design=_build_design_config(namespace),
        alpha_grid=_build_alpha_grid_config(namespace),
        inference=_build_ch_inference_config(namespace),
    )


def build_post_selection_run_config(namespace: Any) -> PostSelectionRunConfig:
    return PostSelectionRunConfig(
        execution=_build_execution_config(namespace),
        design=_build_design_config(namespace),
        alpha_grid=_build_alpha_grid_config(namespace),
        inference=_build_ch_inference_config(namespace),
        selection=PostSelectionConfig(
            selection_lasso_multiplier=float(
                _namespace_value(
                    namespace,
                    "selection_lasso_multiplier",
                    DEFAULT_SELECTION_LASSO_MULTIPLIER,
                )
            )
        ),
    )


def build_dml_run_config(namespace: Any) -> DMLRunConfig:
    return DMLRunConfig(
        execution=_build_execution_config(namespace),
        design=_build_design_config(namespace),
        alpha_grid=_build_alpha_grid_config(namespace),
        dml=DMLConfig(
            k_folds=int(
                _namespace_value(namespace, "dml_k_folds", DEFAULT_DML_K_FOLDS)
            ),
            quantile_penalty=float(
                _namespace_value(
                    namespace,
                    "dml_quantile_penalty",
                    DEFAULT_DML_QUANTILE_PENALTY,
                )
            ),
            ridge_alpha=float(
                _namespace_value(
                    namespace, "dml_ridge_alpha", DEFAULT_DML_RIDGE_ALPHA
                )
            ),
            quantile_solver=str(
                _namespace_value(
                    namespace,
                    "dml_quantile_solver",
                    DEFAULT_DML_QUANTILE_SOLVER,
                )
            ),
            critical_value_multiplier=float(
                _namespace_value(
                    namespace,
                    "critical_value_multiplier",
                    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
                )
            ),
        ),
    )


def build_estimator_run_config(namespace: Any) -> EstimatorRunConfig:
    estimators = tuple(getattr(namespace, "estimators", ()))
    if len(estimators) != 1:
        raise ValueError("exactly one estimator is required to build a run config")
    builders = {
        "oracle": build_oracle_run_config,
        "post_selection": build_post_selection_run_config,
        "dml": build_dml_run_config,
    }
    try:
        builder = builders[estimators[0]]
    except KeyError as exc:
        raise ValueError(f"unknown estimator for run config: {estimators[0]}") from exc
    return builder(namespace)


def runner_kwargs(config: EstimatorRunConfig) -> dict[str, object]:
    """Translate a typed estimator config to the existing runner interface."""
    if isinstance(config, (OracleRunConfig, PostSelectionRunConfig)):
        inference = config.inference
        values: dict[str, object] = {
            "quantreg_max_iter": inference.quantreg_max_iter,
            "critical_value_multiplier": inference.critical_value_multiplier,
            "show_quantreg_warnings": inference.show_quantreg_warnings,
            "grid_strategy": inference.grid_strategy,
            "refinement_tolerance": inference.refinement_tolerance,
            "max_refinement_depth": inference.max_refinement_depth,
            "max_alpha_evaluations": inference.max_alpha_evaluations,
            "iteration_warning_policy": inference.iteration_warning_policy,
            "hard_failure_policy": inference.hard_failure_policy,
            "adaptive_midpoint_probe": inference.adaptive_midpoint_probe,
            "alpha_hat_grid": inference.alpha_hat_grid,
        }
        if isinstance(config, PostSelectionRunConfig):
            values["selection_lasso_multiplier"] = (
                config.selection.selection_lasso_multiplier
            )
        return values
    return {
        "dml_k_folds": config.dml.k_folds,
        "dml_quantile_penalty": config.dml.quantile_penalty,
        "dml_ridge_alpha": config.dml.ridge_alpha,
        "dml_quantile_solver": config.dml.quantile_solver,
        "critical_value_multiplier": config.dml.critical_value_multiplier,
    }


__all__ = [
    "AlphaGridConfig",
    "CHInferenceConfig",
    "DEFAULT_ALPHA_GRID_SIZE",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_ALPHA_MIN",
    "DEFAULT_BASE_SEED",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CRITICAL_VALUE_MULTIPLIER",
    "DEFAULT_DML_K_FOLDS",
    "DEFAULT_DML_QUANTILE_PENALTY",
    "DEFAULT_DML_QUANTILE_SOLVER",
    "DEFAULT_DML_RIDGE_ALPHA",
    "DEFAULT_ESTIMATORS",
    "DEFAULT_N_JOBS",
    "DEFAULT_GRID_STRATEGY",
    "DEFAULT_REFINEMENT_TOLERANCE",
    "DEFAULT_MAX_REFINEMENT_DEPTH",
    "DEFAULT_MAX_ALPHA_EVALUATIONS",
    "DEFAULT_ITERATION_WARNING_POLICY",
    "DEFAULT_HARD_FAILURE_POLICY",
    "DEFAULT_ADAPTIVE_MIDPOINT_PROBE",
    "DEFAULT_ALPHA_HAT_GRID",
    "DEFAULT_QUANTREG_MAX_ITER",
    "DEFAULT_SELECTION_LASSO_MULTIPLIER",
    "DMLConfig",
    "DMLRunConfig",
    "DF_T",
    "DGPS",
    "ESTIMATORS",
    "EstimatorRunConfig",
    "ExecutionConfig",
    "FAST_OUTPUT",
    "N_VALUES",
    "OracleRunConfig",
    "PI_VALUES",
    "P_VALUES",
    "PostSelectionConfig",
    "PostSelectionRunConfig",
    "R_FAST",
    "RHO_UV",
    "RHO_X",
    "TAUS",
    "DesignConfig",
    "build_dml_run_config",
    "build_estimator_run_config",
    "build_oracle_run_config",
    "build_post_selection_run_config",
    "runner_kwargs",
]
