from __future__ import annotations

from dataclasses import dataclass


class DynamicRFError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DynamicRFResult:
    pair_count: int
    mean_shape_distance: float
    mean_log_gain_shift: float
    shape_distance_ci: tuple[float, float]
    gain_shift_ci: tuple[float, float]
    reset_shape_distance: float
    recovery_shape_distances: tuple[float, ...]
    finite_difference_relative_error: float
    teacher_shape_error: float | None
    teacher_gain_error: float | None
    per_source_shape_distances: tuple[float, ...]
    per_source_gain_shifts: tuple[float, ...]
    status: str
    teacher_primary_errors: tuple[float, ...] = ()
    teacher_recovery_errors: tuple[float, ...] = ()
    teacher_gain_direction_agreement: tuple[bool, ...] = ()
    teacher_model_signed_gains: tuple[float, ...] = ()
    teacher_reference_signed_gains: tuple[float, ...] = ()
    teacher_signed_gain_correlation: float | None = None
    teacher_delta_cosine_distance: float | None = None


def classify_dynamic_rf(
    pair_count: int,
    shape_distance: float,
    log_gain_shift: float,
    *,
    identifiable: bool = True,
    reset_shape_distance: float = 0.0,
) -> str:
    if pair_count < 3 or not identifiable:
        return "not_identifiable"
    if reset_shape_distance >= max(shape_distance, 1e-8):
        return "not_supported"
    return (
        "supported"
        if shape_distance > 1e-3 or log_gain_shift > 1e-3
        else "not_supported"
    )


def empty_dynamic_rf_result() -> DynamicRFResult:
    return DynamicRFResult(
        0,
        0.0,
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
        0.0,
        (),
        float("inf"),
        None,
        None,
        (),
        (),
        "not_identifiable",
    )


__all__ = [
    "DynamicRFError",
    "DynamicRFResult",
    "classify_dynamic_rf",
    "empty_dynamic_rf_result",
]
