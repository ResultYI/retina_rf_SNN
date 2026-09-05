from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.mechanistic_retina.mechanism_identifiability import MechanismTeacher
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class SampledCondition:
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainingArrayRequest:
    teacher: MechanismTeacher
    data: MechanisticSeedData
    sampled: SampledCondition | None


@dataclass(frozen=True, slots=True)
class TrainingArrays:
    train_cones: torch.Tensor
    train_observed: torch.Tensor
    train_target: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_observed: torch.Tensor
    validation_target: torch.Tensor
    validation_mask: torch.Tensor


def build_training_arrays(request: TrainingArrayRequest) -> TrainingArrays:
    if request.sampled is None:
        return TrainingArrays(
            request.data.train_cones,
            torch.zeros_like(request.teacher.train_probability),
            request.teacher.train_probability,
            request.data.train_mask[:, 0],
            request.data.validation_cones,
            torch.zeros_like(request.teacher.validation_probability),
            request.teacher.validation_probability,
            request.data.validation_mask[:, 0],
        )
    train_trials = request.sampled.train_spikes.shape[1]
    validation_trials = request.sampled.validation_spikes.shape[1]
    train_cones = request.data.train_cones[:, None].expand(
        -1, train_trials, -1, -1
    ).flatten(0, 1)
    validation_cones = request.data.validation_cones[:, None].expand(
        -1, validation_trials, -1, -1
    ).flatten(0, 1)
    train_spikes = request.sampled.train_spikes.flatten(0, 1)
    validation_spikes = request.sampled.validation_spikes.flatten(0, 1)
    validation_target = request.teacher.validation_probability[:, None].expand_as(
        request.sampled.validation_spikes
    ).flatten(0, 1)
    return TrainingArrays(
        train_cones,
        train_spikes,
        train_spikes,
        request.data.train_mask[:, 0, None]
        .expand_as(request.sampled.train_spikes)
        .flatten(0, 1),
        validation_cones,
        validation_spikes,
        validation_target,
        request.data.validation_mask[:, 0, None]
        .expand_as(request.sampled.validation_spikes)
        .flatten(0, 1),
    )


def bias_ce(arrays: TrainingArrays) -> float:
    count = arrays.train_mask.sum(dim=(0, 1)).clamp_min(1)
    rate = (arrays.train_target * arrays.train_mask).sum(dim=(0, 1)) / count
    logits = torch.logit(rate.clamp(1e-6, 1 - 1e-6)).view(1, 1, -1)
    return float(
        expected_bernoulli_nll(
            logits.expand_as(arrays.validation_target),
            arrays.validation_target,
            arrays.validation_mask,
        )
    )


__all__ = [
    "SampledCondition",
    "TrainingArrayRequest",
    "TrainingArrays",
    "bias_ce",
    "build_training_arrays",
]
