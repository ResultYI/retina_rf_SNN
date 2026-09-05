from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll


@dataclass(frozen=True, slots=True)
class TrainingContract:
    initialization: str
    steps: int
    checkpoint_steps: tuple[int, ...]
    rf_target_loss_used: bool


class TrainingConfig(Protocol):
    steps: int
    checkpoints: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NoiseFreeTrainingRequest:
    model: MechanisticGraphTemporalRetina
    train_cones: torch.Tensor
    train_observed_counts: torch.Tensor
    train_target_probability: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_observed_counts: torch.Tensor
    validation_target_probability: torch.Tensor
    validation_mask: torch.Tensor
    clamps: frozenset[PathwayClamp]
    steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float
    batch_size: int
    seed: int
    progress: Callable[[int, float], None] | None = None


@dataclass(frozen=True, slots=True)
class NoiseFreeTrainingPoint:
    step: int
    train_ce: float
    validation_ce: float


@dataclass(frozen=True, slots=True)
class NoiseFreeTrainingResult:
    checkpoints: tuple[NoiseFreeTrainingPoint, ...]
    gradients_finite: bool


@dataclass(frozen=True, slots=True)
class StructuralTrainingError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    model: MechanisticGraphTemporalRetina
    cones: torch.Tensor
    observed_counts: torch.Tensor
    target_probability: torch.Tensor
    mask: torch.Tensor
    clamps: frozenset[PathwayClamp] = frozenset()


def training_contract(config: TrainingConfig) -> TrainingContract:
    return TrainingContract(
        "teacher-independent-raw",
        config.steps,
        config.checkpoints,
        False,
    )


def train_noise_free(request: NoiseFreeTrainingRequest) -> NoiseFreeTrainingResult:
    _validate(request)
    torch.manual_seed(request.seed)
    parameters = _trainable_parameters(request.model, request.clamps)
    optimizer = torch.optim.Adam(parameters, lr=request.learning_rate)
    generator = torch.Generator().manual_seed(request.seed + 2000)
    points = [_point(request, 0, _train_ce(request, slice(0, 1)))]
    gradients_finite = True
    train_ce = points[0].train_ce
    for step in range(1, request.steps + 1):
        indices = torch.randint(
            request.train_cones.shape[0],
            (request.batch_size,),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = request.model.forward_sequence(
            request.train_cones[indices],
            observed_counts=request.train_observed_counts[indices],
            clamps=request.clamps,
        ).logits
        loss = expected_bernoulli_nll(
            logits,
            request.train_target_probability[indices],
            request.train_mask[indices],
        )
        loss.backward()
        gradients_finite = gradients_finite and _finite_gradients(parameters)
        optimizer.step()
        request.model.project_mechanism_parameters()
        train_ce = float(loss.detach())
        if step in request.checkpoint_steps:
            point = _point(request, step, train_ce)
            points.append(point)
            if request.progress is not None:
                request.progress(step, point.validation_ce)
    return NoiseFreeTrainingResult(tuple(points), gradients_finite)


def validation_ce(
    request: ValidationRequest,
) -> float:
    with torch.no_grad():
        logits = request.model.forward_sequence(
            request.cones,
            observed_counts=request.observed_counts,
            clamps=request.clamps,
        ).logits
        return float(
            expected_bernoulli_nll(logits, request.target_probability, request.mask)
        )


def _point(
    request: NoiseFreeTrainingRequest,
    step: int,
    train_ce: float,
) -> NoiseFreeTrainingPoint:
    return NoiseFreeTrainingPoint(
        step,
        train_ce,
        validation_ce(ValidationRequest(
            request.model,
            request.validation_cones,
            request.validation_observed_counts,
            request.validation_target_probability,
            request.validation_mask,
            request.clamps,
        )),
    )


def _train_ce(request: NoiseFreeTrainingRequest, indices: slice) -> float:
    return validation_ce(ValidationRequest(
        request.model,
        request.train_cones[indices],
        request.train_observed_counts[indices],
        request.train_target_probability[indices],
        request.train_mask[indices],
        request.clamps,
    ))


def _trainable_parameters(
    model: MechanisticGraphTemporalRetina,
    clamps: frozenset[PathwayClamp],
) -> tuple[nn.Parameter, ...]:
    excluded = set()
    if PathwayClamp.H1 in clamps:
        excluded.add("gates.raw_h1_amplitude")
    if PathwayClamp.AMACRINE_LOCAL in clamps:
        excluded.add("gates.ac_local")
    if PathwayClamp.AMACRINE_TRANSIENT in clamps:
        excluded.add("gates.ac_transient")
    if PathwayClamp.RGC_HISTORY in clamps:
        excluded.add("gates.history")
    prefixes = ("bipolar.", "amacrine.", "rgc.response_bias", "gates.", "shared_subunits.")
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(prefixes)
        and name not in excluded
    )


def _finite_gradients(parameters: Sequence[nn.Parameter]) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def _validate(request: NoiseFreeTrainingRequest) -> None:
    if request.steps != 400 or request.checkpoint_steps != (0, 50, 100, 200, 400):
        raise StructuralTrainingError("mechanism training must use the fixed 400-step schedule")
    if not (
        request.train_observed_counts.shape
        == request.train_target_probability.shape
        == request.train_mask.shape
    ):
        raise StructuralTrainingError("training likelihood tensor shapes differ")
    if not (
        request.validation_observed_counts.shape
        == request.validation_target_probability.shape
        == request.validation_mask.shape
    ):
        raise StructuralTrainingError("validation likelihood tensor shapes differ")


__all__ = [
    "NoiseFreeTrainingRequest",
    "NoiseFreeTrainingResult",
    "StructuralTrainingError",
    "TrainingContract",
    "ValidationRequest",
    "train_noise_free",
    "training_contract",
    "validation_ce",
]
