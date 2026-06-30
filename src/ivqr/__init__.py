"""IVQR grid, moment, and confidence-region helpers."""

__all__ = [
    "ConfidenceRegion",
    "alpha_grid",
    "critical_value_chi_square",
    "estimate_ch_ivqr_controls",
    "evaluate_alpha_ch_ivqr",
    "invert_score_test",
]


def __getattr__(name: str):
    if name == "alpha_grid":
        from ivqr.alpha_grid import alpha_grid

        return alpha_grid
    if name in {"evaluate_alpha_ch_ivqr", "estimate_ch_ivqr_controls"}:
        from ivqr import ch_inverse

        return getattr(ch_inverse, name)
    if name in {"ConfidenceRegion", "critical_value_chi_square", "invert_score_test"}:
        from ivqr import confidence_regions

        return getattr(confidence_regions, name)
    raise AttributeError(name)
