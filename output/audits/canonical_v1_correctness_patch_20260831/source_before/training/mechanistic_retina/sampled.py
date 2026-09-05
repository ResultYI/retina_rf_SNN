from __future__ import annotations

from dataclasses import dataclass

import torch

from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters


@dataclass(frozen=True, slots=True)
class SampledTrainingRequest:
    model: MechanisticGraphTemporalRetina
    train_cones: torch.Tensor
    train_spikes: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_spikes: torch.Tensor
    validation_mask: torch.Tensor
    validation_probability: torch.Tensor
    steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float
    batch_size: int
    seed: int


@dataclass(frozen=True, slots=True)
class SampledTrainingPoint:
    step: int
    train_nll: float
    expected_validation_ce: float
    sampled_validation_nll: float
    validation_logit_rmse: float


@dataclass(frozen=True, slots=True)
class SampledTrainingResult:
    checkpoints: tuple[SampledTrainingPoint, ...]
    gradients_finite: bool


@dataclass(frozen=True, slots=True)
class SampledTrainingError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def train_sampled_model(request: SampledTrainingRequest) -> SampledTrainingResult:
    _validate(request)
    torch.manual_seed(request.seed)
    optimizer = build_phase1_optimizer(request.model, learning_rate=request.learning_rate)
    generator = torch.Generator().manual_seed(request.seed + 1000)
    points = [_point(request, 0, _initial_train_nll(request))]
    gradients_finite = True
    train_nll = points[0].train_nll
    for step in range(1, request.steps + 1):
        stimuli = torch.randint(
            request.train_spikes.shape[0], (request.batch_size,), generator=generator
        )
        trials = torch.randint(
            request.train_spikes.shape[1], (request.batch_size,), generator=generator
        )
        cones = request.train_cones[stimuli]
        counts = request.train_spikes[stimuli, trials]
        mask = request.train_mask[stimuli, trials]
        optimizer.zero_grad(set_to_none=True)
        logits = request.model.forward_sequence(cones, observed_counts=counts).logits
        loss = expected_bernoulli_nll(logits, counts, mask)
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in phase1_parameters(request.model)
        )
        optimizer.step()
        request.model.project_mechanism_parameters()
        train_nll = float(loss.detach())
        if step in request.checkpoint_steps:
            points.append(_point(request, step, train_nll))
    return SampledTrainingResult(tuple(points), gradients_finite)


def predict_sampled_model(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    spikes: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    stimuli, trials = spikes.shape[:2]
    flat_cones = cones[:, None].expand(-1, trials, -1, -1).flatten(0, 1)
    flat_spikes = spikes.flatten(0, 1)
    chunks = []
    with torch.no_grad():
        for start in range(0, flat_spikes.shape[0], chunk_size):
            stop = start + chunk_size
            chunks.append(
                model.forward_sequence(
                    flat_cones[start:stop], observed_counts=flat_spikes[start:stop]
                ).logits
            )
    return torch.cat(chunks).reshape(stimuli, trials, *flat_spikes.shape[1:])


def _point(
    request: SampledTrainingRequest,
    step: int,
    train_nll: float,
) -> SampledTrainingPoint:
    logits = predict_sampled_model(
        request.model, request.validation_cones, request.validation_spikes
    )
    expected = request.validation_probability[:, None].expand_as(logits)
    sampled = expected_bernoulli_nll(logits, request.validation_spikes, request.validation_mask)
    expected_ce = expected_bernoulli_nll(logits, expected, request.validation_mask)
    target_logits = torch.logit(expected.clamp(1e-7, 1 - 1e-7))
    mask = request.validation_mask.to(dtype=logits.dtype)
    rmse = (((logits - target_logits).square() * mask).sum() / mask.sum()).sqrt()
    return SampledTrainingPoint(step, train_nll, float(expected_ce), float(sampled), float(rmse))


def _initial_train_nll(request: SampledTrainingRequest) -> float:
    logits = request.model.forward_sequence(
        request.train_cones[:1], observed_counts=request.train_spikes[:1, 0]
    ).logits
    return float(expected_bernoulli_nll(logits, request.train_spikes[:1, 0], request.train_mask[:1, 0]))


def _validate(request: SampledTrainingRequest) -> None:
    train_shape = request.train_spikes.shape
    validation_shape = request.validation_spikes.shape
    if request.train_mask.shape != train_shape or request.validation_mask.shape != validation_shape:
        raise SampledTrainingError("sampled spikes and masks must share shapes")
    if request.train_cones.shape[0] != train_shape[0] or request.validation_cones.shape[0] != validation_shape[0]:
        raise SampledTrainingError("stimulus identities do not match sampled splits")
    if request.validation_probability.shape != validation_shape[0:1] + validation_shape[2:]:
        raise SampledTrainingError("validation probabilities do not match sampled validation")
    if request.checkpoint_steps[-1] != request.steps or 0 not in request.checkpoint_steps:
        raise SampledTrainingError("training checkpoints must include zero and the fixed final step")


__all__ = [
    "SampledTrainingError",
    "SampledTrainingPoint",
    "SampledTrainingRequest",
    "SampledTrainingResult",
    "predict_sampled_model",
    "train_sampled_model",
]
