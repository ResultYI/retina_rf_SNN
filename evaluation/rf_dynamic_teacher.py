from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch.nn import functional as F

_EPS: Final = 1e-8
_DIRECTION_EPS: Final = 1e-6
_PURE_GAIN_SHAPE_LIMIT: Final = 0.05


@dataclass(frozen=True, slots=True)
class TeacherDynamicReference:
    low_kernel: torch.Tensor
    high_kernel: torch.Tensor
    context_gain_envelope: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "low_kernel", torch.flip(self.low_kernel, dims=(1,)))
        object.__setattr__(self, "high_kernel", torch.flip(self.high_kernel, dims=(1,)))


@dataclass(frozen=True, slots=True)
class RecoveryContract:
    delays_ms: tuple[int, ...]
    dt_ms: float


@dataclass(frozen=True, slots=True)
class TeacherDynamicAlignment:
    predicted_signed_gains: tuple[float, ...]
    teacher_signed_gains: tuple[float, ...]
    direction_agreement: tuple[bool, ...]
    signed_gain_mae: float
    signed_gain_correlation: float
    kernel_delta_cosine_distance: float | None
    primary_error: float | None
    excessive_shape_deformation: bool
    identifiable: bool
    status: str


def align_teacher_dynamic_rf(
    predicted_low: torch.Tensor,
    predicted_high: torch.Tensor,
    reference: TeacherDynamicReference,
) -> TeacherDynamicAlignment:
    low, high, teacher_low, teacher_high = _common_shape(
        predicted_low,
        predicted_high,
        reference.low_kernel.to(predicted_low.device),
        reference.high_kernel.to(predicted_low.device),
    )
    predicted_gain = _signed_log_gain(low, high)
    teacher_gain = _signed_log_gain(teacher_low, teacher_high)
    teacher_active = teacher_gain.abs() > _DIRECTION_EPS
    direction = torch.sign(predicted_gain) == torch.sign(teacher_gain)
    gain_mae = (predicted_gain - teacher_gain).abs().mean()
    delta_distance = _delta_cosine_distance(low, high, teacher_low, teacher_high)
    excessive_shape = (
        _shape_distance(teacher_low, teacher_high) <= _DIRECTION_EPS
        and _shape_distance(low, high) > _PURE_GAIN_SHAPE_LIMIT
    )
    if not bool(teacher_active.any()):
        return TeacherDynamicAlignment(
            predicted_signed_gains=tuple(float(value) for value in predicted_gain.cpu()),
            teacher_signed_gains=tuple(float(value) for value in teacher_gain.cpu()),
            direction_agreement=(),
            signed_gain_mae=float(gain_mae),
            signed_gain_correlation=0.0,
            kernel_delta_cosine_distance=None,
            primary_error=None,
            excessive_shape_deformation=excessive_shape,
            identifiable=True,
            status="not_supported",
        )
    identifiable = bool(
        torch.isfinite(predicted_gain).all()
        and torch.isfinite(teacher_gain).all()
        and torch.isfinite(delta_distance)
    )
    status = _alignment_status(identifiable, direction, excessive_shape)
    return TeacherDynamicAlignment(
        predicted_signed_gains=tuple(float(value) for value in predicted_gain.cpu()),
        teacher_signed_gains=tuple(float(value) for value in teacher_gain.cpu()),
        direction_agreement=tuple(bool(value) for value in direction.cpu()),
        signed_gain_mae=float(gain_mae),
        signed_gain_correlation=_gain_correlation(predicted_gain, teacher_gain),
        kernel_delta_cosine_distance=float(delta_distance),
        primary_error=float(gain_mae + delta_distance),
        excessive_shape_deformation=excessive_shape,
        identifiable=identifiable,
        status=status,
    )


def teacher_recovery_errors(
    model_recovery: tuple[
        tuple[tuple[float, ...], ...],
        ...,
    ],
    reference: TeacherDynamicReference,
    contract: RecoveryContract,
) -> tuple[float, ...]:
    if reference.context_gain_envelope is None:
        return ()
    expected = _teacher_recovery(reference.context_gain_envelope, contract)
    if not model_recovery:
        return ()
    count = min(len(model_recovery), len(expected))
    return tuple(
        _curve_error(model_recovery[index], expected[index])
        for index in range(count)
    )


def classify_teacher_status(
    base_status: str,
    alignments: list[TeacherDynamicAlignment],
) -> str:
    if base_status != "supported" or not alignments:
        return base_status
    statuses = {alignment.status for alignment in alignments}
    if "not_identifiable" in statuses:
        return "not_identifiable"
    if "not_supported" in statuses:
        return "not_supported"
    if "teacher_mismatch" in statuses:
        return "teacher_mismatch"
    return "supported"


def mean_tuple(values: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not values:
        return ()
    width = len(values[0])
    return tuple(sum(row[index] for row in values) / len(values) for index in range(width))


def mean_optional(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def all_tuple(values: list[tuple[bool, ...]]) -> tuple[bool, ...]:
    if not values:
        return ()
    width = len(values[0])
    return tuple(all(row[index] for row in values) for index in range(width))


def _common_shape(
    left_low: torch.Tensor,
    left_high: torch.Tensor,
    right_low: torch.Tensor,
    right_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cells = min(left_low.shape[0], right_low.shape[0])
    lags = min(left_low.shape[1], right_low.shape[1])
    cones = min(left_low.shape[2], right_low.shape[2])
    return (
        left_low[:cells, -lags:, :cones],
        left_high[:cells, -lags:, :cones],
        right_low[:cells, -lags:, :cones],
        right_high[:cells, -lags:, :cones],
    )


def _signed_log_gain(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    numerator = high.flatten(1).norm(dim=1) + _EPS
    denominator = low.flatten(1).norm(dim=1) + _EPS
    return (numerator / denominator).log()


def _delta_cosine_distance(
    predicted_low: torch.Tensor,
    predicted_high: torch.Tensor,
    teacher_low: torch.Tensor,
    teacher_high: torch.Tensor,
) -> torch.Tensor:
    predicted = (predicted_high - predicted_low).flatten(1)
    teacher = (teacher_high - teacher_low).flatten(1)
    active = teacher.norm(dim=1) > _DIRECTION_EPS
    if not bool(active.any()):
        return torch.tensor(float("inf"), device=predicted.device)
    cosine = F.cosine_similarity(predicted[active], teacher[active], dim=1)
    return (1 - cosine.clamp(-1, 1)).mean()


def _shape_distance(low: torch.Tensor, high: torch.Tensor) -> float:
    cosine = F.cosine_similarity(low.flatten(1), high.flatten(1), dim=1)
    return float((1 - cosine.clamp(-1, 1)).mean())


def _gain_correlation(predicted: torch.Tensor, teacher: torch.Tensor) -> float:
    predicted_centered = predicted - predicted.mean()
    teacher_centered = teacher - teacher.mean()
    denominator = predicted_centered.norm() * teacher_centered.norm()
    if denominator <= _EPS:
        return 0.0
    return float((predicted_centered * teacher_centered).sum() / denominator)


def _alignment_status(
    identifiable: bool,
    direction: torch.Tensor,
    excessive_shape: bool,
) -> str:
    if not identifiable:
        return "not_identifiable"
    if not bool(direction.all()) or excessive_shape:
        return "teacher_mismatch"
    return "supported"


def _teacher_recovery(
    envelope: torch.Tensor,
    contract: RecoveryContract,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    low = envelope[0::2]
    high = envelope[1::2]
    if low.shape[1] < 2:
        return ()
    previous_effect = high[:, -2] - low[:, -2]
    final_effect = high[:, -1] - low[:, -1]
    ratio = (final_effect.abs() / previous_effect.abs().clamp_min(_EPS)).clamp(0, 1)
    return tuple(
        tuple(
            tuple(
                float(value)
                for value in (
                    (
                        1
                        + (high[source, -1] - 1)
                        * ratio[source]
                        ** max(0, round(delay_ms / contract.dt_ms))
                    ).clamp_min(_EPS)
                    / (
                        1
                        + (low[source, -1] - 1)
                        * ratio[source]
                        ** max(0, round(delay_ms / contract.dt_ms))
                    ).clamp_min(_EPS)
                ).log()
            )
            for delay_ms in contract.delays_ms
        )
        for source in range(low.shape[0])
    )


def _curve_error(
    model_curve: tuple[tuple[float, ...], ...],
    expected_curve: tuple[tuple[float, ...], ...],
) -> float:
    delays = min(len(model_curve), len(expected_curve))
    if delays == 0:
        return float("inf")
    errors = [
        abs(model_curve[delay][cell] - expected_curve[delay][cell])
        for delay in range(delays)
        for cell in range(min(len(model_curve[delay]), len(expected_curve[delay])))
    ]
    return sum(errors) / len(errors) if errors else float("inf")


__all__ = [
    "RecoveryContract",
    "TeacherDynamicAlignment",
    "TeacherDynamicReference",
    "align_teacher_dynamic_rf",
    "all_tuple",
    "classify_teacher_status",
    "mean_optional",
    "mean_tuple",
    "teacher_recovery_errors",
]
