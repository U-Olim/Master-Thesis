"""Simulation constants for the IVQR Monte Carlo study."""

N_VALUES: tuple[int, ...] = (500, 1000)
P_VALUES: tuple[int, ...] = (200, 500)
PI_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)
DGPS: tuple[str, ...] = ("dgp1", "dgp2", "dgp3")

ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")
DEFAULT_ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")

R_FAST: int = 10
R_FULL: int = 500

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
FULL_OUTPUT: str = "results/raw/full_results.csv"


__all__ = [
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
    "DF_T",
    "DGPS",
    "ESTIMATORS",
    "FAST_OUTPUT",
    "FULL_OUTPUT",
    "N_VALUES",
    "PI_VALUES",
    "P_VALUES",
    "R_FAST",
    "R_FULL",
    "RHO_UV",
    "RHO_X",
    "TAUS",
]
