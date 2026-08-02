from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from evaluation.rf_dynamic_metrics import context_pairs
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit, masked_history_counts


class TypeGainOracleError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TypeGainOracleResult:
    max_iterations: int
    evaluations: int
    losses: tuple[float, ...]
    before_cell_gains: tuple[float, ...]
    after_cell_gains: tuple[float, ...]
    teacher_cell_gains: tuple[float, ...]
    before_direction_count: int
    after_direction_count: int
    before_mae: float
    after_mae: float
    max_abs_error: float
    converged: bool


@dataclass(frozen=True, slots=True)
class GainEvaluationRequest:
    model: ResponseRetinaModel
    split: ResponseSplit
    cell_type_ids: tuple[str, ...]
    lag_steps: int


@dataclass(frozen=True, slots=True)
class GainOracleRequest:
    evaluation: GainEvaluationRequest
    target_cell_gains: torch.Tensor
    named_parameters: tuple[tuple[str, nn.Parameter], ...]
    max_iterations: int = 40
    tolerance: float = 1e-3


def differentiable_type_gains(
    request: GainEvaluationRequest,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = request.model
    split = request.split
    device = next(model.parameters()).device
    stimulus_count, trial_count, time_count, cell_count = split.spike_counts.shape
    if cell_count != len(request.cell_type_ids) or not 1 <= request.lag_steps <= time_count:
        raise TypeGainOracleError("RF gain tensor dimensions are invalid")
    stimuli = split.cone_response[:, None].expand(
        -1,
        trial_count,
        -1,
        -1,
    ).reshape(stimulus_count * trial_count, *split.cone_response.shape[1:])
    stimuli = stimuli.to(device).detach().clone().requires_grad_(True)
    counts = split.spike_counts.reshape(
        stimulus_count * trial_count,
        time_count,
        cell_count,
    ).to(device)
    mask = split.valid_mask.reshape_as(split.spike_counts).reshape_as(counts).to(device)
    model.eval()
    output, _ = model.forward_sequence(
        stimuli,
        observed_counts=masked_history_counts(counts, mask),
    )
    kernels = []
    for cell in range(cell_count):
        gradient = torch.autograd.grad(
            output.spike_logits[:, -1, cell].sum(),
            stimuli,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        kernels.append(gradient[:, -request.lag_steps:])
    by_stimulus = torch.stack(kernels, dim=1).reshape(
        stimulus_count,
        trial_count,
        cell_count,
        request.lag_steps,
        split.cone_response.shape[2],
    ).mean(dim=1)
    pair_gains = []
    for low, high in context_pairs(split):
        low_norm = by_stimulus[low].flatten(1).norm(dim=1)
        high_norm = by_stimulus[high].flatten(1).norm(dim=1)
        pair_gains.append(((high_norm + 1e-8) / (low_norm + 1e-8)).log())
    if not pair_gains:
        raise TypeGainOracleError("No matched context pair was found")
    cell_gains = torch.stack(pair_gains).mean(dim=0)
    return cell_gains, group_means(cell_gains, request.cell_type_ids)


def group_means(
    cell_gains: torch.Tensor,
    cell_type_ids: tuple[str, ...],
) -> torch.Tensor:
    values = []
    for type_id in ("midget", "parasol"):
        indices = [index for index, value in enumerate(cell_type_ids) if value == type_id]
        if not indices:
            raise TypeGainOracleError(f"Missing {type_id} cells")
        values.append(cell_gains[indices].mean())
    return torch.stack(values)


def run_gain_oracle(request: GainOracleRequest) -> TypeGainOracleResult:
    if not request.named_parameters or request.max_iterations < 0 or request.tolerance <= 0:
        raise TypeGainOracleError("Oracle parameters, iterations, and tolerance are invalid")
    evaluation = request.evaluation
    target = request.target_cell_gains.to(next(evaluation.model.parameters()).device)
    if target.shape != (len(evaluation.cell_type_ids),):
        raise TypeGainOracleError("Oracle target must contain one gain per cell")
    parameters = tuple(parameter for _, parameter in request.named_parameters)
    original_values = tuple(parameter.detach().clone() for parameter in parameters)
    original_trainable = tuple(
        parameter.requires_grad for parameter in evaluation.model.parameters()
    )
    losses: list[float] = []
    try:
        before, _ = differentiable_type_gains(
            evaluation,
            create_graph=False,
        )
        for parameter in evaluation.model.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)

        if request.max_iterations:
            optimizer = torch.optim.LBFGS(
                parameters,
                max_iter=request.max_iterations,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                cell_gains, _ = differentiable_type_gains(
                    evaluation,
                    create_graph=True,
                )
                loss = torch.nn.functional.mse_loss(cell_gains, target)
                if not torch.isfinite(loss):
                    raise TypeGainOracleError("Oracle objective became non-finite")
                loss.backward()
                losses.append(float(loss.detach()))
                return loss

            optimizer.step(closure)

        after, _ = differentiable_type_gains(
            evaluation,
            create_graph=False,
        )
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, original_values, strict=True):
                parameter.copy_(value)
        for parameter, trainable in zip(
            evaluation.model.parameters(),
            original_trainable,
            strict=True,
        ):
            parameter.requires_grad_(trainable)
    error = (after - target).abs()
    return TypeGainOracleResult(
        max_iterations=request.max_iterations,
        evaluations=len(losses),
        losses=tuple(losses),
        before_cell_gains=tuple(float(value) for value in before),
        after_cell_gains=tuple(float(value) for value in after.detach()),
        teacher_cell_gains=tuple(float(value) for value in target.detach()),
        before_direction_count=_direction_count(before, target),
        after_direction_count=_direction_count(after, target),
        before_mae=float((before - target).abs().mean()),
        after_mae=float(error.mean()),
        max_abs_error=float(error.max()),
        converged=bool(error.max() <= request.tolerance),
    )


def _direction_count(values: torch.Tensor, target: torch.Tensor) -> int:
    return int((torch.sign(values) == torch.sign(target)).sum())


__all__ = [
    "GainEvaluationRequest",
    "GainOracleRequest",
    "TypeGainOracleError",
    "TypeGainOracleResult",
    "differentiable_type_gains",
    "group_means",
    "run_gain_oracle",
]
