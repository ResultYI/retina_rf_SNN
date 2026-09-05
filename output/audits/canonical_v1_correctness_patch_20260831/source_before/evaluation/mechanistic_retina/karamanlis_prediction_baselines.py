from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics

import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.karamanlis_glm_math import (
    build_static_flash_design,
    local_static_flash_logits,
)
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.real_sampled import (
    SpikePredictionMetrics,
    spike_prediction_metrics,
)


@dataclass(frozen=True, slots=True)
class PopulationBaselineError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class PopulationGLMTrainingRequest:
    train: RealSequenceSplit
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    graph_radius_deg: float | None
    temporal_lags: int
    steps: int
    seed: int
    support_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class PopulationGLMTrainingResult:
    model: LocalPointProcessGLM
    gradients_finite: bool
    actually_updated: tuple[str, ...]
    train_nll_initial: float
    train_nll_trained: float
    solver_iterations: int
    converged: bool


def constant_rate_logits(
    train_events: torch.Tensor,
    train_mask: torch.Tensor,
    evaluation_events: torch.Tensor,
    evaluation_mask: torch.Tensor,
) -> torch.Tensor:
    _require_aligned(train_events, train_mask)
    _require_aligned(evaluation_events, evaluation_mask)
    if train_events.shape[-1] != evaluation_events.shape[-1]:
        raise PopulationBaselineError("train and evaluation cell counts differ")
    float_mask = train_mask.to(dtype=train_events.dtype)
    denominator = float_mask.sum(dim=(0, 1)).clamp_min(1)
    rates = (train_events * float_mask).sum(dim=(0, 1)) / denominator
    bias = torch.logit(rates.clamp(1e-5, 1 - 1e-5))
    return bias.view(1, 1, -1).expand_as(evaluation_events).clone()


def fit_population_glm(
    request: PopulationGLMTrainingRequest,
) -> PopulationGLMTrainingResult:
    if request.steps < 1:
        raise PopulationBaselineError("GLM solver iterations must be positive")
    torch.manual_seed(request.seed)
    model = LocalPointProcessGLM(
        request.cone_positions,
        request.cell_positions,
        request.graph_radius_deg,
        request.temporal_lags,
        support_mask=request.support_mask,
    )
    bias = constant_rate_logits(
        request.train.spike_events,
        request.train.valid_mask,
        request.train.spike_events[:1, :1],
        request.train.valid_mask[:1, :1],
    )[0, 0]
    with torch.no_grad():
        model.bias.copy_(bias)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    design = build_static_flash_design(request.train)
    events = request.train.spike_events
    mask = request.train.valid_mask
    history = events * mask.to(dtype=events.dtype)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=request.steps,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        logits = local_static_flash_logits(model, design, history)
        return expected_bernoulli_nll(logits, events, mask)

    initial_nll = float(objective().detach())

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_loss = objective()
    final_loss.backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    gradients_finite = len(gradients) > 0 and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = max(float(gradient.abs().max()) for gradient in gradients)
    first_parameter = next(iter(model.parameters()))
    iterations = int(
        optimizer.state[first_parameter].get("n_iter", request.steps)
    )
    updated = tuple(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(initial[name], parameter.detach())
    )
    return PopulationGLMTrainingResult(
        model,
        gradients_finite,
        updated,
        initial_nll,
        float(final_loss.detach()),
        iterations,
        gradients_finite and gradient_norm <= 1e-4,
    )


def evaluate_population_glm(
    model: LocalPointProcessGLM,
    split: RealSequenceSplit,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    design = build_static_flash_design(split)
    model.eval()
    with torch.no_grad():
        history = split.spike_events * split.valid_mask.to(
            dtype=split.spike_events.dtype
        )
        logits = local_static_flash_logits(model, design, history)
    return spike_prediction_metrics(logits, split.spike_events, split.valid_mask), logits


def evaluate_retinal_model(
    model: MechanisticGraphTemporalRetina,
    split: RealSequenceSplit,
    *,
    chunk_size: int = 8,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, split.cone_drive.shape[0], chunk_size):
            stop = start + chunk_size
            rows.append(
                model.forward_sequence(
                    split.cone_drive[start:stop],
                    observed_counts=split.spike_events[start:stop],
                ).logits
            )
    logits = torch.cat(rows)
    return spike_prediction_metrics(logits, split.spike_events, split.valid_mask), logits


def winner_counts(
    per_model: Mapping[str, Sequence[float]],
) -> Mapping[str, int]:
    names, cell_count = _model_shape(per_model)
    counts = {name: 0 for name in names}
    for cell in range(cell_count):
        winner = min(names, key=lambda name: per_model[name][cell])
        counts[winner] += 1
    return counts


def grouped_nll(
    per_model: Mapping[str, Sequence[float]],
    polarities: Sequence[str],
    cell_types: Sequence[str],
) -> Mapping[str, Mapping[str, float]]:
    names, cell_count = _model_shape(per_model)
    if len(polarities) != cell_count or len(cell_types) != cell_count:
        raise PopulationBaselineError("cell metadata and NLL lengths differ")
    labels = tuple(
        f"{polarity} {cell_type}"
        for polarity, cell_type in zip(polarities, cell_types, strict=True)
    )
    return {
        label: {
            name: statistics.fmean(
                per_model[name][index]
                for index, current in enumerate(labels)
                if current == label
            )
            for name in names
        }
        for label in dict.fromkeys(labels)
    }


def _require_aligned(values: torch.Tensor, mask: torch.Tensor) -> None:
    if values.ndim != 3 or values.shape != mask.shape:
        raise PopulationBaselineError("event values and valid mask must align")


def _model_shape(
    per_model: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], int]:
    names = tuple(per_model)
    if not names:
        raise PopulationBaselineError("at least one model is required")
    cell_count = len(per_model[names[0]])
    if cell_count < 1 or any(len(per_model[name]) != cell_count for name in names):
        raise PopulationBaselineError("per-model NLL lengths must align")
    return names, cell_count


__all__ = [
    "PopulationBaselineError",
    "PopulationGLMTrainingRequest",
    "PopulationGLMTrainingResult",
    "constant_rate_logits",
    "evaluate_population_glm",
    "evaluate_retinal_model",
    "fit_population_glm",
    "grouped_nll",
    "winner_counts",
]
