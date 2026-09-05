from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    constant_rate_logits,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.real_sampled import (
    SpikePredictionMetrics,
    spike_prediction_metrics,
)


@dataclass(frozen=True, slots=True)
class DynamicGLMTrainingRequest:
    train: RealSequenceSplit
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    temporal_lags: int
    history_lags: int
    max_iterations: int
    seed: int
    l2_penalty: float = 1e-4


@dataclass(frozen=True, slots=True)
class DynamicGLMTrainingResult:
    model: LocalPointProcessGLM
    gradients_finite: bool
    actually_updated: tuple[str, ...]
    train_nll_initial: float
    train_nll_trained: float
    solver_iterations: int
    solver_evaluations: int
    final_gradient_max: float
    strict_gradient_converged: bool
    solver_terminated_before_budget: bool
    converged: bool


@dataclass(frozen=True, slots=True)
class DynamicGLMError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def fit_dynamic_glm(
    request: DynamicGLMTrainingRequest,
) -> DynamicGLMTrainingResult:
    if request.max_iterations < 1:
        raise DynamicGLMError("GLM solver iterations must be positive")
    if not math.isfinite(request.l2_penalty) or request.l2_penalty < 0:
        raise DynamicGLMError("GLM L2 penalty must be finite and nonnegative")
    if request.train.spike_events.shape[-1] != 1:
        raise DynamicGLMError("cell-wise dynamic GLM requires exactly one cell")
    torch.manual_seed(request.seed)
    support = torch.ones(
        (1, request.cone_positions.shape[0]),
        dtype=torch.bool,
        device=request.cone_positions.device,
    )
    model = LocalPointProcessGLM(
        request.cone_positions,
        request.cell_positions,
        radius_deg=None,
        temporal_lags=request.temporal_lags,
        history_lags=request.history_lags,
        support_mask=support,
    )
    train = request.train
    bias = constant_rate_logits(
        train.spike_events,
        train.valid_mask,
        train.spike_events[:1, :1],
        train.valid_mask[:1, :1],
    )[0, 0]
    with torch.no_grad():
        model.bias.copy_(bias)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=request.max_iterations,
        max_eval=request.max_iterations * 10,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def data_nll() -> torch.Tensor:
        logits = model(train.cone_drive, train.spike_events)
        return expected_bernoulli_nll(
            logits, train.spike_events, train.valid_mask
        )

    def objective() -> torch.Tensor:
        penalty = sum(
            parameter.square().sum()
            for name, parameter in model.named_parameters()
            if name != "bias"
        )
        return data_nll() + request.l2_penalty * penalty

    initial_nll = float(data_nll().detach())

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_objective = objective()
    final_objective.backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    gradient_max = (
        max(float(gradient.abs().max()) for gradient in gradients)
        if gradients
        else float("inf")
    )
    first_parameter = next(iter(model.parameters()))
    iterations = int(
        optimizer.state[first_parameter].get("n_iter", request.max_iterations)
    )
    evaluations = int(
        optimizer.state[first_parameter].get(
            "func_evals", request.max_iterations * 10
        )
    )
    updated = tuple(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(initial[name], parameter.detach())
    )
    strict_gradient_converged = gradient_max <= 1e-4
    terminated_before_budget = (
        iterations < request.max_iterations
        and evaluations < request.max_iterations * 10
    )
    return DynamicGLMTrainingResult(
        model=model,
        gradients_finite=gradients_finite,
        actually_updated=updated,
        train_nll_initial=initial_nll,
        train_nll_trained=float(data_nll().detach()),
        solver_iterations=iterations,
        solver_evaluations=evaluations,
        final_gradient_max=gradient_max,
        strict_gradient_converged=strict_gradient_converged,
        solver_terminated_before_budget=terminated_before_budget,
        converged=gradients_finite
        and (strict_gradient_converged or terminated_before_budget),
    )


def evaluate_dynamic_glm(
    model: LocalPointProcessGLM,
    split: RealSequenceSplit,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(split.cone_drive, split.spike_events)
    return (
        spike_prediction_metrics(logits, split.spike_events, split.valid_mask),
        logits,
    )


__all__ = [
    "DynamicGLMError",
    "DynamicGLMTrainingRequest",
    "DynamicGLMTrainingResult",
    "evaluate_dynamic_glm",
    "fit_dynamic_glm",
]
