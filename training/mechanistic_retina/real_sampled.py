from __future__ import annotations

from dataclasses import dataclass

import torch

from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import (
    build_phase1_optimizer,
    phase1_parameters,
)


@dataclass(frozen=True, slots=True)
class RealSpikeTrainingRequest:
    model: MechanisticGraphTemporalRetina
    train: RealSequenceSplit
    validation: RealSequenceSplit
    steps: int
    learning_rate: float
    batch_size: int
    seed: int


@dataclass(frozen=True, slots=True)
class RealSpikeTrainingResult:
    validation_nll_raw: float
    validation_nll_trained: float
    mean_probability_raw: float
    mean_probability_trained: float
    observed_event_rate: float
    per_cell_nll_raw: tuple[float, ...]
    per_cell_nll_trained: tuple[float, ...]
    per_cell_probability_raw: tuple[float, ...]
    per_cell_probability_trained: tuple[float, ...]
    per_cell_event_rate: tuple[float, ...]
    gradients_finite: bool
    actually_updated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpikePredictionMetrics:
    population_nll: float
    mean_probability: float
    observed_event_rate: float
    per_cell_nll: tuple[float, ...]
    per_cell_probability: tuple[float, ...]
    per_cell_event_rate: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RealSpikeTrainingError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def fit_real_spike_model(
    request: RealSpikeTrainingRequest,
) -> RealSpikeTrainingResult:
    _validate(request)
    torch.manual_seed(request.seed)
    optimizer = build_phase1_optimizer(
        request.model, learning_rate=request.learning_rate
    )
    parameters = phase1_parameters(request.model)
    names = {
        id(parameter): name for name, parameter in request.model.named_parameters()
    }
    initial = tuple(parameter.detach().clone() for parameter in parameters)
    raw_metrics = _validation_metrics(
        request.model, request.validation
    )
    generator = torch.Generator().manual_seed(request.seed + 1_000_003)
    gradients_finite = True
    request.model.train()
    for _ in range(request.steps):
        indices = torch.randint(
            request.train.cone_drive.shape[0],
            (request.batch_size,),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = request.model.forward_sequence(
            request.train.cone_drive[indices],
            observed_counts=request.train.spike_events[indices],
        ).logits
        loss = expected_bernoulli_nll(
            logits,
            request.train.spike_events[indices],
            request.train.valid_mask[indices],
        )
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        optimizer.step()
        request.model.project_mechanism_parameters()
    trained_metrics = _validation_metrics(
        request.model, request.validation
    )
    updated = tuple(
        names[id(parameter)]
        for before, parameter in zip(initial, parameters, strict=True)
        if not torch.equal(before, parameter.detach())
    )
    return RealSpikeTrainingResult(
        validation_nll_raw=raw_metrics.population_nll,
        validation_nll_trained=trained_metrics.population_nll,
        mean_probability_raw=raw_metrics.mean_probability,
        mean_probability_trained=trained_metrics.mean_probability,
        observed_event_rate=raw_metrics.observed_event_rate,
        per_cell_nll_raw=raw_metrics.per_cell_nll,
        per_cell_nll_trained=trained_metrics.per_cell_nll,
        per_cell_probability_raw=raw_metrics.per_cell_probability,
        per_cell_probability_trained=trained_metrics.per_cell_probability,
        per_cell_event_rate=raw_metrics.per_cell_event_rate,
        gradients_finite=gradients_finite,
        actually_updated=updated,
    )


def _validation_metrics(
    model: MechanisticGraphTemporalRetina,
    split: RealSequenceSplit,
) -> SpikePredictionMetrics:
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, split.cone_drive.shape[0], 8):
            stop = start + 8
            output = model.forward_sequence(
                split.cone_drive[start:stop],
                observed_counts=split.spike_events[start:stop],
            )
            logits.append(output.logits)
    return spike_prediction_metrics(
        torch.cat(logits), split.spike_events, split.valid_mask
    )


def spike_prediction_metrics(
    logits: torch.Tensor,
    spike_events: torch.Tensor,
    valid_mask: torch.Tensor,
) -> SpikePredictionMetrics:
    if logits.shape != spike_events.shape or valid_mask.shape != logits.shape:
        raise RealSpikeTrainingError("prediction tensors must share one shape")
    per_cell_nll = tuple(
        float(expected_bernoulli_nll(logits[..., cell], spike_events[..., cell], valid_mask[..., cell]))
        for cell in range(logits.shape[-1])
    )
    probabilities = torch.sigmoid(logits)
    mask = valid_mask.to(dtype=logits.dtype)
    denominators = mask.sum(dim=(0, 1)).clamp_min(1)
    per_cell_probability = tuple(
        float(value) for value in ((probabilities * mask).sum(dim=(0, 1)) / denominators)
    )
    per_cell_event_rate = tuple(
        float(value) for value in ((spike_events * mask).sum(dim=(0, 1)) / denominators)
    )
    return SpikePredictionMetrics(
        population_nll=float(expected_bernoulli_nll(logits, spike_events, valid_mask)),
        mean_probability=float((probabilities * mask).sum() / mask.sum().clamp_min(1)),
        observed_event_rate=float((spike_events * mask).sum() / mask.sum().clamp_min(1)),
        per_cell_nll=per_cell_nll,
        per_cell_probability=per_cell_probability,
        per_cell_event_rate=per_cell_event_rate,
    )


def _validate(request: RealSpikeTrainingRequest) -> None:
    if request.steps < 1 or request.batch_size < 1 or request.learning_rate <= 0:
        raise RealSpikeTrainingError("training hyperparameters must be positive")
    for split in (request.train, request.validation):
        expected = split.spike_events.shape
        if split.spike_counts.shape != expected or split.valid_mask.shape != expected:
            raise RealSpikeTrainingError("real spike tensors must share one shape")
        if split.cone_drive.shape[:2] != expected[:2]:
            raise RealSpikeTrainingError("real stimulus and spike time axes differ")
        if not bool(
            torch.all((split.spike_events == 0) | (split.spike_events == 1))
        ):
            raise RealSpikeTrainingError("real spike-event targets must be binary")


__all__ = [
    "RealSpikeTrainingError",
    "RealSpikeTrainingRequest",
    "RealSpikeTrainingResult",
    "SpikePredictionMetrics",
    "fit_real_spike_model",
    "spike_prediction_metrics",
]
