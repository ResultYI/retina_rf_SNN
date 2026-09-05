from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import nn

from baselines.point_process_glm import PointProcessGLM
from evaluation.model_comparison.prediction import LogitFunction, fit_bias, masked_bernoulli_loss
from evaluation.model_comparison.types import TrainingPoint


ProgressCallback = Callable[[int, float], None]


@dataclass(frozen=True, slots=True)
class BaselineTrainingRequest:
    model: nn.Module
    logits: LogitFunction
    train_cones: torch.Tensor
    train_spikes: torch.Tensor
    train_mask: torch.Tensor
    steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float
    batch_size: int
    seed: int
    progress: ProgressCallback | None = None


@dataclass(frozen=True, slots=True)
class BaselineTrainingResult:
    checkpoints: tuple[TrainingPoint, ...]
    gradients_finite: bool
    converged: bool


def initialize_bias(model: nn.Module, spikes: torch.Tensor, mask: torch.Tensor) -> None:
    parameter = getattr(model, "bias", None)
    if not isinstance(parameter, nn.Parameter):
        raise ValueError("baseline model has no trainable bias")
    with torch.no_grad():
        parameter.copy_(fit_bias(spikes, mask))


def train_baseline(request: BaselineTrainingRequest) -> BaselineTrainingResult:
    if request.checkpoint_steps[0] != 0 or request.checkpoint_steps[-1] != request.steps:
        raise ValueError("baseline checkpoints must include zero and final step")
    optimizer = torch.optim.Adam(request.model.parameters(), lr=request.learning_rate)
    generator = torch.Generator().manual_seed(request.seed + 4000)
    initial = _batch_loss(request, torch.arange(min(request.batch_size, request.train_spikes.shape[0])), torch.zeros(min(request.batch_size, request.train_spikes.shape[0]), dtype=torch.long))
    points = [TrainingPoint(0, float(initial), 0.0)]
    gradients_finite = True
    recent = [float(initial)]
    for step in range(1, request.steps + 1):
        stimuli = torch.randint(
            request.train_spikes.shape[0], (request.batch_size,), generator=generator
        )
        trials = torch.randint(
            request.train_spikes.shape[1], (request.batch_size,), generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(request, stimuli, trials)
        loss.backward()
        parameters = tuple(request.model.parameters())
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        gradients_finite = gradients_finite and finite
        gradient_norm = max(
            float(parameter.grad.detach().abs().max())
            for parameter in parameters
            if parameter.grad is not None
        )
        optimizer.step()
        recent.append(float(loss.detach()))
        if len(recent) > 50:
            recent.pop(0)
        if step in request.checkpoint_steps:
            point = TrainingPoint(step, float(loss.detach()), gradient_norm)
            points.append(point)
            if request.progress is not None:
                request.progress(step, point.train_nll)
    relative_change = abs(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
    return BaselineTrainingResult(
        tuple(points), gradients_finite, gradients_finite and relative_change <= 0.05
    )


def train_glm_lbfgs(request: BaselineTrainingRequest) -> BaselineTrainingResult:
    if not isinstance(request.model, PointProcessGLM):
        raise ValueError("L-BFGS solver requires PointProcessGLM")
    stimulus_count, trial_count = request.train_spikes.shape[:2]
    cones = request.train_cones[:, None].expand(
        -1, trial_count, -1, -1
    ).reshape(stimulus_count * trial_count, *request.train_cones.shape[1:])
    counts = request.train_spikes.flatten(0, 1)
    mask = request.train_mask.flatten(0, 1)
    optimizer = torch.optim.LBFGS(
        request.model.parameters(),
        lr=1.0,
        max_iter=request.steps,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        return masked_bernoulli_loss(request.logits(cones, counts), counts, mask)

    initial = float(objective().detach())

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final = objective()
    final.backward()
    gradients = tuple(
        parameter.grad
        for parameter in request.model.parameters()
        if parameter.grad is not None
    )
    finite = len(gradients) > 0 and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_norm = max(float(gradient.detach().abs().max()) for gradient in gradients)
    first_parameter = next(iter(request.model.parameters()))
    iterations = int(optimizer.state[first_parameter].get("n_iter", request.steps))
    point = TrainingPoint(iterations, float(final.detach()), gradient_norm)
    if request.progress is not None:
        request.progress(iterations, point.train_nll)
    return BaselineTrainingResult(
        (TrainingPoint(0, initial, 0.0), point),
        finite,
        finite and gradient_norm <= 1e-4,
    )


def _batch_loss(
    request: BaselineTrainingRequest,
    stimuli: torch.Tensor,
    trials: torch.Tensor,
) -> torch.Tensor:
    counts = request.train_spikes[stimuli, trials]
    mask = request.train_mask[stimuli, trials]
    logits = request.logits(request.train_cones[stimuli], counts)
    return masked_bernoulli_loss(logits, counts, mask)


__all__ = [
    "BaselineTrainingRequest",
    "BaselineTrainingResult",
    "ProgressCallback",
    "initialize_bias",
    "train_baseline",
    "train_glm_lbfgs",
]
