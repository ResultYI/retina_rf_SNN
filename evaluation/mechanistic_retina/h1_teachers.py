from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.candidate0_likelihood_math import causal_static_drive
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


@dataclass(frozen=True, slots=True)
class H1TeacherRequest:
    model: MechanisticGraphTemporalRetina
    train_cones: torch.Tensor
    validation_cones: torch.Tensor
    base_rf: torch.Tensor
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    response_bias: float


@dataclass(frozen=True, slots=True)
class MatchedH1Teachers:
    present_train_probability: torch.Tensor
    present_validation_probability: torch.Tensor
    absent_train_probability: torch.Tensor
    absent_validation_probability: torch.Tensor
    present_bias: torch.Tensor
    absent_bias: torch.Tensor
    present_mean_rate: float
    absent_mean_rate: float
    teacher_h1_component_rf: torch.Tensor


@dataclass(frozen=True, slots=True)
class H1TeacherError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def build_matched_h1_teachers(request: H1TeacherRequest) -> MatchedH1Teachers:
    _validate(request)
    absent_train_drive = causal_static_drive(request.train_cones, request.base_rf)
    absent_validation_drive = causal_static_drive(request.validation_cones, request.base_rf)
    with torch.no_grad():
        present_train_cones = request.model.h1(request.train_cones, clamped=False).modulated_cones
        present_validation_cones = request.model.h1(
            request.validation_cones, clamped=False
        ).modulated_cones
    present_train_drive = causal_static_drive(present_train_cones, request.base_rf)
    present_validation_drive = causal_static_drive(present_validation_cones, request.base_rf)
    mean, deviation = _calibration(absent_train_drive, request.train_mask)
    absent_bias = torch.full_like(mean, request.response_bias)
    absent_train_logits = _standardize(absent_train_drive, mean, deviation) + absent_bias
    absent_validation_logits = _standardize(absent_validation_drive, mean, deviation) + absent_bias
    target_rate = _masked_rate(torch.sigmoid(absent_train_logits), request.train_mask)
    present_train_standard = _standardize(present_train_drive, mean, deviation)
    present_bias = _matched_bias(present_train_standard, request.train_mask, target_rate)
    present_train_logits = present_train_standard + present_bias
    present_validation_logits = _standardize(present_validation_drive, mean, deviation) + present_bias
    component = _h1_component_rf(request, mean, deviation)
    present_train = torch.sigmoid(present_train_logits)
    absent_train = torch.sigmoid(absent_train_logits)
    return MatchedH1Teachers(
        present_train,
        torch.sigmoid(present_validation_logits),
        absent_train,
        torch.sigmoid(absent_validation_logits),
        present_bias,
        absent_bias,
        float(_masked_rate(present_train, request.train_mask).mean()),
        float(_masked_rate(absent_train, request.train_mask).mean()),
        component,
    )


def _h1_component_rf(
    request: H1TeacherRequest,
    mean: torch.Tensor,
    deviation: torch.Tensor,
) -> torch.Tensor:
    stimulus = request.validation_cones.detach().clone().requires_grad_(True)
    present = causal_static_drive(
        request.model.h1(stimulus, clamped=False).modulated_cones, request.base_rf
    )
    absent = causal_static_drive(stimulus, request.base_rf)
    difference = (present - absent) / deviation
    kernels = []
    for cell in range(difference.shape[-1]):
        gradient = torch.autograd.grad(
            difference[:, -1, cell].sum(),
            stimulus,
            retain_graph=cell + 1 < difference.shape[-1],
        )[0]
        kernels.append(gradient[:, -request.model.config.lag_steps :])
    return torch.stack(kernels, dim=1).detach()


def _calibration(drive: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=(0, 1)).clamp_min(1)
    mean = (drive * mask).sum(dim=(0, 1)) / count
    centered = drive - mean
    deviation = ((centered.square() * mask).sum(dim=(0, 1)) / count).sqrt().clamp_min(1e-6)
    return mean, deviation


def _standardize(drive: torch.Tensor, mean: torch.Tensor, deviation: torch.Tensor) -> torch.Tensor:
    return (drive - mean.view(1, 1, -1)) / deviation.view(1, 1, -1)


def _masked_rate(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (probability * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)


def _matched_bias(
    standardized: torch.Tensor,
    mask: torch.Tensor,
    target_rate: torch.Tensor,
) -> torch.Tensor:
    low = torch.full_like(target_rate, -12.0)
    high = torch.full_like(target_rate, 12.0)
    for _ in range(64):
        middle = (low + high) * 0.5
        rate = _masked_rate(torch.sigmoid(standardized + middle), mask)
        low = torch.where(rate < target_rate, middle, low)
        high = torch.where(rate >= target_rate, middle, high)
    return (low + high) * 0.5


def _validate(request: H1TeacherRequest) -> None:
    if request.train_mask.shape != request.train_cones.shape[:2] + request.base_rf.shape[:1]:
        raise H1TeacherError("H1 train identities do not match base RF")
    if request.validation_mask.shape != request.validation_cones.shape[:2] + request.base_rf.shape[:1]:
        raise H1TeacherError("H1 validation identities do not match base RF")
    if request.base_rf.shape[1] != request.model.config.lag_steps:
        raise H1TeacherError("H1 teacher lag count differs from frozen model")


__all__ = [
    "H1TeacherError",
    "H1TeacherRequest",
    "MatchedH1Teachers",
    "build_matched_h1_teachers",
]
