"""Estimator-name normalization for simulation scenarios."""

from __future__ import annotations

from collections.abc import Sequence


CANONICAL_ESTIMATORS: tuple[str, ...] = (
    "oracle",
    "dml",
    "post_selection",
    "full_control",
)
MAIN_SCENARIO_ESTIMATORS: tuple[str, ...] = (
    "oracle",
    "dml",
    "post_selection",
)
FULL_CONTROL_SCENARIO_ESTIMATORS: tuple[str, ...] = ("full_control",)

ESTIMATOR_ALIASES: dict[str, str] = {
    "oracle": "oracle",
    "dml": "dml",
    "dml_ivqr": "dml",
    "dml-ivqr": "dml",
    "post_selection": "post_selection",
    "post_selection_ivqr": "post_selection",
    "post-selection": "post_selection",
    "post-selection-ivqr": "post_selection",
    "postselection": "post_selection",
    "full_control": "full_control",
    "full_control_ivqr": "full_control",
    "full-control": "full_control",
    "full-control-ivqr": "full_control",
}

SCENARIO_DEFAULT_ESTIMATORS: dict[str, tuple[str, ...]] = {
    "main": MAIN_SCENARIO_ESTIMATORS,
    "full_control": FULL_CONTROL_SCENARIO_ESTIMATORS,
}


def _normalize_token(raw_estimator: str) -> str:
    token = raw_estimator.strip().lower().replace(" ", "_")
    return token.replace("-", "_") if token not in ESTIMATOR_ALIASES else token


def normalize_estimator_names(
    raw_estimators: Sequence[str] | None,
    *,
    scenario: str,
) -> tuple[str, ...]:
    """Normalize, deduplicate, and validate estimator names for a scenario."""
    if scenario not in SCENARIO_DEFAULT_ESTIMATORS:
        valid_scenarios = ", ".join(sorted(SCENARIO_DEFAULT_ESTIMATORS))
        raise ValueError(
            f"Unknown estimator scenario {scenario!r}. "
            f"Valid scenarios: {valid_scenarios}"
        )

    scenario_estimators = SCENARIO_DEFAULT_ESTIMATORS[scenario]
    if raw_estimators is None:
        return scenario_estimators
    if isinstance(raw_estimators, str):
        raise ValueError("estimators must be a sequence of estimator names")

    normalized: list[str] = []
    invalid: list[str] = []
    for raw in raw_estimators:
        if not isinstance(raw, str) or not raw.strip():
            invalid.append(str(raw))
            continue
        token = _normalize_token(raw)
        canonical = ESTIMATOR_ALIASES.get(token)
        if canonical is None:
            invalid.append(raw)
            continue
        if canonical not in normalized:
            normalized.append(canonical)

    if invalid:
        valid = ", ".join(CANONICAL_ESTIMATORS)
        aliases = ", ".join(sorted(ESTIMATOR_ALIASES))
        raise ValueError(
            f"Unknown estimator name(s): {invalid}. "
            f"Valid estimators: {valid}. Supported aliases: {aliases}."
        )
    if not normalized:
        raise ValueError("estimators must contain at least one estimator name")

    unsupported = [name for name in normalized if name not in scenario_estimators]
    if unsupported:
        valid = ", ".join(scenario_estimators)
        raise ValueError(
            f"Estimator(s) {unsupported} are not supported for scenario {scenario!r}. "
            f"Valid choices: {valid}."
        )

    return tuple(normalized)


__all__ = [
    "CANONICAL_ESTIMATORS",
    "ESTIMATOR_ALIASES",
    "FULL_CONTROL_SCENARIO_ESTIMATORS",
    "MAIN_SCENARIO_ESTIMATORS",
    "SCENARIO_DEFAULT_ESTIMATORS",
    "normalize_estimator_names",
]
