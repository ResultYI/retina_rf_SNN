from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.rf_dynamic_metrics import (
    bootstrap_ci,
    context_pairs,
    kernel_metrics,
    recovery_distance,
    reset_distance,
    teacher_errors,
)
from evaluation.rf_static import extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


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


@dataclass(frozen=True, slots=True)
class DynamicRFComparison:
    pair_count: int
    learned_shape_delta: float
    learned_gain_delta: float
    shape_delta_ci: tuple[float, float]
    gain_delta_ci: tuple[float, float]
    status: str


def evaluate_dynamic_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    *,
    lag_steps: int,
    recovery_delays_ms: tuple[int, ...] = (0,),
    dt_ms: float = 5.0,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
    teacher_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> DynamicRFResult:
    pairs = context_pairs(split)
    if not pairs:
        return _empty_result()
    shapes: list[float] = []
    gains: list[float] = []
    numerical_errors: list[float] = []
    identifiable = True
    sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = next(model.parameters()).device
    for low_index, high_index in pairs:
        low = split.cone_response[low_index : low_index + 1].to(device)
        high = split.cone_response[high_index : high_index + 1].to(device)
        if not torch.equal(low[:, -lag_steps:], high[:, -lag_steps:]):
            raise DynamicRFError(
                "Dynamic RF context pairs need an identical final probe"
            )
        low_rf = extract_static_rf(model, low, lag_steps=lag_steps)
        high_rf = extract_static_rf(model, high, lag_steps=lag_steps)
        shape, gain = kernel_metrics(low_rf.kernels, high_rf.kernels)
        shapes.append(shape)
        gains.append(gain)
        numerical_errors.extend(
            (
                low_rf.finite_difference_relative_error,
                high_rf.finite_difference_relative_error,
            )
        )
        identifiable = identifiable and low_rf.identifiable and high_rf.identifiable
        sequences.append((low, high))
    reset_shape = reset_distance(model, sequences[0], lag_steps)
    recovery = tuple(
        recovery_distance(model, sequences, lag_steps, delay, dt_ms)
        for delay in recovery_delays_ms
    )
    shape_ci = bootstrap_ci(shapes, bootstrap_iterations, seed)
    gain_ci = bootstrap_ci(gains, bootstrap_iterations, seed + 1)
    teacher_shape_error, teacher_gain_error = teacher_errors(
        shapes,
        gains,
        teacher_kernels,
    )
    mean_shape = sum(shapes) / len(shapes)
    mean_gain = sum(gains) / len(gains)
    finite_error = max(numerical_errors)
    status = classify_dynamic_rf(
        len(pairs),
        mean_shape,
        mean_gain,
        identifiable=identifiable,
        reset_shape_distance=reset_shape,
    )
    return DynamicRFResult(
        pair_count=len(pairs),
        mean_shape_distance=mean_shape,
        mean_log_gain_shift=mean_gain,
        shape_distance_ci=shape_ci,
        gain_shift_ci=gain_ci,
        reset_shape_distance=reset_shape,
        recovery_shape_distances=recovery,
        finite_difference_relative_error=finite_error,
        teacher_shape_error=teacher_shape_error,
        teacher_gain_error=teacher_gain_error,
        per_source_shape_distances=tuple(shapes),
        per_source_gain_shifts=tuple(gains),
        status=status,
    )


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


def compare_dynamic_rf(
    trained: DynamicRFResult,
    initialized: DynamicRFResult,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> DynamicRFComparison:
    if trained.pair_count != initialized.pair_count:
        raise DynamicRFError("Trained and initialized RF pairs must match")
    shape_deltas = [
        trained_value - initialized_value
        for trained_value, initialized_value in zip(
            trained.per_source_shape_distances,
            initialized.per_source_shape_distances,
            strict=True,
        )
    ]
    gain_deltas = [
        trained_value - initialized_value
        for trained_value, initialized_value in zip(
            trained.per_source_gain_shifts,
            initialized.per_source_gain_shifts,
            strict=True,
        )
    ]
    if not shape_deltas:
        return DynamicRFComparison(
            0,
            0.0,
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            "not_identifiable",
        )
    shape_ci = bootstrap_ci(shape_deltas, bootstrap_iterations, seed)
    gain_ci = bootstrap_ci(gain_deltas, bootstrap_iterations, seed + 1)
    status = (
        "supported"
        if trained.status == "supported"
        and (shape_ci[0] > 0 or gain_ci[0] > 0)
        else "not_supported"
    )
    return DynamicRFComparison(
        trained.pair_count,
        sum(shape_deltas) / len(shape_deltas),
        sum(gain_deltas) / len(gain_deltas),
        shape_ci,
        gain_ci,
        status,
    )


def _empty_result() -> DynamicRFResult:
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
    "DynamicRFComparison",
    "DynamicRFError",
    "DynamicRFResult",
    "classify_dynamic_rf",
    "compare_dynamic_rf",
    "evaluate_dynamic_rf",
]
