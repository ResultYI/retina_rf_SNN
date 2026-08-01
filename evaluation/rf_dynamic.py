from __future__ import annotations

import torch

from evaluation.rf_dynamic_metrics import (
    bootstrap_ci,
    context_pairs,
    kernel_metrics,
    teacher_errors,
    trial_conditioned_rf,
)
from evaluation.rf_dynamic_recovery import (
    mean_distances,
    recovery_distances_by_source,
    reset_distance,
)
from evaluation.rf_dynamic_compare import DynamicRFComparison, compare_dynamic_rf
from evaluation.rf_dynamic_result import (
    DynamicRFError,
    DynamicRFResult,
    classify_dynamic_rf,
    empty_dynamic_rf_result,
)
from evaluation.rf_dynamic_teacher import (
    RecoveryContract,
    TeacherDynamicAlignment,
    TeacherDynamicReference,
    align_teacher_dynamic_rf,
    all_tuple,
    classify_teacher_status,
    mean_optional,
    mean_tuple,
    teacher_recovery_errors,
)
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


def evaluate_dynamic_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    *,
    lag_steps: int,
    condition_on_observed: bool = True,
    recovery_delays_ms: tuple[int, ...] = (0,),
    dt_ms: float = 5.0,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
    teacher_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
    teacher_context_gain_envelope: torch.Tensor | None = None,
) -> DynamicRFResult:
    pairs = context_pairs(split)
    if not pairs:
        return empty_dynamic_rf_result()
    shapes: list[float] = []
    gains: list[float] = []
    numerical_errors: list[float] = []
    low_kernels: list[torch.Tensor] = []
    high_kernels: list[torch.Tensor] = []
    identifiable = True
    teacher_alignments: list[TeacherDynamicAlignment] = []
    teacher_reference = (
        TeacherDynamicReference(
            teacher_kernels[0],
            teacher_kernels[1],
            teacher_context_gain_envelope,
        )
        if teacher_kernels is not None
        else None
    )
    device = next(model.parameters()).device
    for low_index, high_index in pairs:
        low = split.cone_response[low_index : low_index + 1].to(device)
        high = split.cone_response[high_index : high_index + 1].to(device)
        if not torch.equal(low[:, -lag_steps:], high[:, -lag_steps:]):
            raise DynamicRFError(
                "Dynamic RF context pairs need an identical final probe"
            )
        low_rf = trial_conditioned_rf(
            model,
            split,
            low_index,
            lag_steps,
            condition_on_observed=condition_on_observed,
        )
        high_rf = trial_conditioned_rf(
            model,
            split,
            high_index,
            lag_steps,
            condition_on_observed=condition_on_observed,
        )
        shape, gain = kernel_metrics(low_rf.kernels, high_rf.kernels)
        low_kernels.append(low_rf.kernels)
        high_kernels.append(high_rf.kernels)
        if teacher_reference is not None:
            teacher_alignments.append(
                align_teacher_dynamic_rf(
                    low_rf.kernels,
                    high_rf.kernels,
                    teacher_reference,
                )
            )
        shapes.append(shape)
        gains.append(gain)
        numerical_errors.extend(
            (
                low_rf.finite_difference_relative_error,
                high_rf.finite_difference_relative_error,
            )
        )
        identifiable = identifiable and low_rf.identifiable and high_rf.identifiable
    reset_by_source = tuple(
        reset_distance(
            model,
            split,
            pair,
            lag_steps,
            condition_on_observed=condition_on_observed,
        )
        for pair in pairs
    )
    reset = mean_distances(reset_by_source)
    recovery_by_source = recovery_distances_by_source(
        model,
        split,
        pairs,
        lag_steps,
        recovery_delays_ms,
        dt_ms,
        condition_on_observed=condition_on_observed,
    )
    recovery = tuple(
        mean_distances(
            tuple(source_curve[delay_index] for source_curve in recovery_by_source)
        )
        for delay_index in range(len(recovery_delays_ms))
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
        reset_shape_distance=reset.shape_distance,
        reset_log_gain_shift=reset.mean_absolute_gain_shift,
    )
    if teacher_reference is not None:
        status = classify_teacher_status(status, teacher_alignments)
    recovery_errors = (
        teacher_recovery_errors(
            tuple(
                tuple(point.signed_gain_shifts for point in source_curve)
                for source_curve in recovery_by_source
            ),
            teacher_reference,
            RecoveryContract(recovery_delays_ms, dt_ms),
        )
        if teacher_reference is not None
        else ()
    )
    primary_errors = tuple(
        alignment.primary_error
        for alignment in teacher_alignments
        if alignment.primary_error is not None
    )
    return DynamicRFResult(
        pair_count=len(pairs),
        mean_shape_distance=mean_shape,
        mean_log_gain_shift=mean_gain,
        shape_distance_ci=shape_ci,
        gain_shift_ci=gain_ci,
        reset_shape_distance=reset.shape_distance,
        recovery_shape_distances=tuple(
            point.shape_distance for point in recovery
        ),
        finite_difference_relative_error=finite_error,
        teacher_shape_error=teacher_shape_error,
        teacher_gain_error=teacher_gain_error,
        per_source_shape_distances=tuple(shapes),
        per_source_gain_shifts=tuple(gains),
        status=status,
        teacher_primary_errors=primary_errors,
        teacher_recovery_errors=recovery_errors,
        teacher_gain_direction_agreement=all_tuple(
            [alignment.direction_agreement for alignment in teacher_alignments]
        ),
        teacher_model_signed_gains=mean_tuple(
            [alignment.predicted_signed_gains for alignment in teacher_alignments]
        ),
        teacher_reference_signed_gains=mean_tuple(
            [alignment.teacher_signed_gains for alignment in teacher_alignments]
        ),
        teacher_signed_gain_correlation=mean_optional(
            [alignment.signed_gain_correlation for alignment in teacher_alignments]
        ),
        teacher_delta_cosine_distance=mean_optional(
            [
                alignment.kernel_delta_cosine_distance
                for alignment in teacher_alignments
                if alignment.kernel_delta_cosine_distance is not None
            ]
        ),
        reset_log_gain_shift=reset.mean_absolute_gain_shift,
        recovery_mean_log_gain_shifts=tuple(
            point.mean_absolute_gain_shift for point in recovery
        ),
        recovery_signed_gain_shifts=tuple(
            point.signed_gain_shifts for point in recovery
        ),
        per_source_reset_shape_distances=tuple(
            point.shape_distance for point in reset_by_source
        ),
        per_source_reset_gain_shifts=tuple(
            point.mean_absolute_gain_shift for point in reset_by_source
        ),
        mean_low_kernel=torch.stack(low_kernels).mean(dim=0).detach().cpu(),
        mean_high_kernel=torch.stack(high_kernels).mean(dim=0).detach().cpu(),
    )

__all__ = [
    "DynamicRFComparison",
    "DynamicRFError",
    "DynamicRFResult",
    "classify_dynamic_rf",
    "compare_dynamic_rf",
    "evaluate_dynamic_rf",
]
