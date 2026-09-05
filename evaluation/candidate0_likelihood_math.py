from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class StaticTargetRequest:
    train_drive: torch.Tensor
    validation_drive: torch.Tensor
    train_mask: torch.Tensor
    validation_trial_count: int
    burn_in_steps: int
    response_bias: float


@dataclass(frozen=True, slots=True)
class StaticTeacherTargets:
    train_probabilities: torch.Tensor
    validation_probabilities: torch.Tensor
    train_logits: torch.Tensor
    validation_logits: torch.Tensor
    drive_mean: torch.Tensor
    drive_std: torch.Tensor


@dataclass(frozen=True, slots=True)
class SupportAuditRequest:
    initial_weights: torch.Tensor
    gradient: torch.Tensor
    projection_weights: torch.Tensor
    threshold: float


@dataclass(frozen=True, slots=True)
class ProjectionSupportAudit:
    positive_support_count: int
    positive_zero_count: int
    positive_zero_feasible_fraction: float
    parameter_direction_cosine: float
    gradient_finite: bool


@dataclass(frozen=True, slots=True)
class LikelihoodMathError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def causal_static_drive(cones: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    if cones.ndim != 3 or rf.ndim != 3 or cones.shape[-1] != rf.shape[-1]:
        raise LikelihoodMathError("static drive requires [stimulus,time,cone] and [cell,lag,cone]")
    lag_steps = rf.shape[1]
    padded = F.pad(cones, (0, 0, lag_steps - 1, 0))
    lagged = padded.unfold(1, lag_steps, 1).permute(0, 1, 3, 2)
    return torch.einsum("stlc,rlc->str", lagged, rf)


def build_static_teacher_targets(
    request: StaticTargetRequest,
) -> StaticTeacherTargets:
    if request.train_mask.ndim != 4 or request.validation_trial_count < 1:
        raise LikelihoodMathError("static target mask and trial count are invalid")
    train_trials = request.train_mask.shape[1]
    train_values = request.train_drive.unsqueeze(1).expand(-1, train_trials, -1, -1)
    valid = request.train_mask.clone()
    valid[:, :, : request.burn_in_steps] = False
    dimensions = (0, 1, 2)
    count = valid.sum(dim=dimensions).clamp_min(1)
    mean = (train_values * valid).sum(dim=dimensions) / count
    centered = train_values - mean.view(1, 1, 1, -1)
    variance = (centered.square() * valid).sum(dim=dimensions) / count
    deviation = variance.sqrt().clamp_min(1e-6)
    train_logits = centered / deviation.view(1, 1, 1, -1) + request.response_bias
    validation = request.validation_drive.unsqueeze(1).expand(
        -1, request.validation_trial_count, -1, -1
    )
    validation_logits = (
        validation - mean.view(1, 1, 1, -1)
    ) / deviation.view(1, 1, 1, -1) + request.response_bias
    tensors = (train_logits, validation_logits, mean, deviation)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise LikelihoodMathError("static teacher calibration produced non-finite values")
    return StaticTeacherTargets(
        torch.sigmoid(train_logits),
        torch.sigmoid(validation_logits),
        train_logits,
        validation_logits,
        mean,
        deviation,
    )


def audit_projection_support(
    request: SupportAuditRequest,
) -> ProjectionSupportAudit:
    if request.initial_weights.shape != request.gradient.shape or (
        request.gradient.shape != request.projection_weights.shape
    ):
        raise LikelihoodMathError("support audit tensors must share shape")
    support = request.projection_weights > request.threshold
    zero = request.initial_weights <= request.threshold
    positive_zero = support & zero
    feasible_zero = positive_zero & (request.gradient < 0)
    feasible_direction = -request.gradient.clone()
    feasible_direction[zero & (request.gradient >= 0)] = 0
    target_direction = request.projection_weights - request.initial_weights
    denominator = feasible_direction.norm() * target_direction.norm()
    cosine = float(
        torch.dot(feasible_direction.flatten(), target_direction.flatten())
        / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    )
    zero_count = int(torch.count_nonzero(positive_zero))
    fraction = int(torch.count_nonzero(feasible_zero)) / max(zero_count, 1)
    return ProjectionSupportAudit(
        int(torch.count_nonzero(support)),
        zero_count,
        fraction,
        cosine,
        bool(torch.isfinite(request.gradient).all()) and math.isfinite(cosine),
    )


__all__ = [
    "LikelihoodMathError",
    "ProjectionSupportAudit",
    "StaticTargetRequest",
    "StaticTeacherTargets",
    "SupportAuditRequest",
    "audit_projection_support",
    "build_static_teacher_targets",
    "causal_static_drive",
]
