from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from evaluation.rf_dynamic_metrics import bootstrap_ci


class DynamicRFLike(Protocol):
    pair_count: int
    per_source_shape_distances: tuple[float, ...]
    per_source_gain_shifts: tuple[float, ...]
    status: str
    teacher_primary_errors: tuple[float, ...]
    teacher_recovery_errors: tuple[float, ...]
    teacher_gain_direction_agreement: tuple[bool, ...]
    teacher_model_signed_gains: tuple[float, ...]
    teacher_reference_signed_gains: tuple[float, ...]


class DynamicRFComparisonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DynamicRFComparison:
    pair_count: int
    learned_shape_delta: float
    learned_gain_delta: float
    shape_delta_ci: tuple[float, float]
    gain_delta_ci: tuple[float, float]
    status: str
    teacher_primary_error_delta: float | None = None
    teacher_primary_error_delta_ci: tuple[float, float] | None = None
    teacher_recovery_error_delta: float | None = None
    teacher_recovery_error_delta_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class TeacherComparisonGate:
    status: str | None
    primary_error_delta: float | None
    primary_error_delta_ci: tuple[float, float] | None
    recovery_error_delta: float | None
    recovery_error_delta_ci: tuple[float, float] | None


def compare_dynamic_rf(
    trained: DynamicRFLike,
    initialized: DynamicRFLike,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> DynamicRFComparison:
    if trained.pair_count != initialized.pair_count:
        raise DynamicRFComparisonError("Trained and initialized RF pairs must match")
    shape_deltas = _deltas(
        trained.per_source_shape_distances,
        initialized.per_source_shape_distances,
    )
    gain_deltas = _deltas(
        trained.per_source_gain_shifts,
        initialized.per_source_gain_shifts,
    )
    if not shape_deltas:
        return DynamicRFComparison(0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), "not_identifiable")
    shape_ci = bootstrap_ci(list(shape_deltas), bootstrap_iterations, seed)
    gain_ci = bootstrap_ci(list(gain_deltas), bootstrap_iterations, seed + 1)
    teacher_gate = _compare_teacher_gate(trained, initialized, bootstrap_iterations, seed + 2)
    status = teacher_gate.status or _unsigned_status(trained.status, shape_ci, gain_ci)
    return DynamicRFComparison(
        trained.pair_count,
        sum(shape_deltas) / len(shape_deltas),
        sum(gain_deltas) / len(gain_deltas),
        shape_ci,
        gain_ci,
        status,
        teacher_gate.primary_error_delta,
        teacher_gate.primary_error_delta_ci,
        teacher_gate.recovery_error_delta,
        teacher_gate.recovery_error_delta_ci,
    )


def _compare_teacher_gate(
    trained: DynamicRFLike,
    initialized: DynamicRFLike,
    bootstrap_iterations: int,
    seed: int,
) -> TeacherComparisonGate:
    if not trained.teacher_primary_errors and not initialized.teacher_primary_errors:
        return TeacherComparisonGate(None, None, None, None, None)
    if trained.pair_count < 3 or trained.status == "not_identifiable":
        return TeacherComparisonGate("not_identifiable", None, None, None, None)
    direction_status = _direction_status(trained)
    if direction_status is not None:
        return TeacherComparisonGate(direction_status, None, None, None, None)
    primary_delta, primary_ci = _error_improvement(
        initialized.teacher_primary_errors,
        trained.teacher_primary_errors,
        trained.pair_count,
        bootstrap_iterations,
        seed,
    )
    recovery_delta, recovery_ci = _error_improvement(
        initialized.teacher_recovery_errors,
        trained.teacher_recovery_errors,
        trained.pair_count,
        bootstrap_iterations,
        seed + 1,
    )
    if trained.status == "teacher_mismatch":
        status = "teacher_mismatch"
    elif primary_ci is None or recovery_ci is None:
        status = "not_supported"
    else:
        status = "supported" if primary_ci[0] > 0 and recovery_ci[0] > 0 else "not_supported"
    return TeacherComparisonGate(status, primary_delta, primary_ci, recovery_delta, recovery_ci)


def _error_improvement(
    initial_errors: tuple[float, ...],
    trained_errors: tuple[float, ...],
    pair_count: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[float | None, tuple[float, float] | None]:
    if (
        len(initial_errors) != pair_count
        or len(trained_errors) != pair_count
        or not _finite_values(initial_errors)
        or not _finite_values(trained_errors)
    ):
        return None, None
    improvements = _deltas(initial_errors, trained_errors)
    return (
        sum(improvements) / len(improvements),
        bootstrap_ci(list(improvements), bootstrap_iterations, seed),
    )


def _deltas(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(left_value - right_value for left_value, right_value in zip(left, right, strict=True))


def _direction_status(result: DynamicRFLike) -> str | None:
    count = len(result.teacher_gain_direction_agreement)
    if (
        count == 0
        or count != len(result.teacher_model_signed_gains)
        or count != len(result.teacher_reference_signed_gains)
        or not _finite_values(result.teacher_model_signed_gains)
        or not _finite_values(result.teacher_reference_signed_gains)
    ):
        return "not_supported"
    if not all(result.teacher_gain_direction_agreement):
        return "teacher_mismatch"
    return None


def _finite_values(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _unsigned_status(
    trained_status: str,
    shape_ci: tuple[float, float],
    gain_ci: tuple[float, float],
) -> str:
    if trained_status != "supported":
        return "not_supported"
    return "supported" if shape_ci[0] > 0 or gain_ci[0] > 0 else "not_supported"


__all__ = [
    "DynamicRFComparison",
    "DynamicRFComparisonError",
    "compare_dynamic_rf",
]
