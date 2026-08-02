from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.parameter_audit import parameter_role
from evaluation.rf_dynamic_metrics import context_pairs
from evaluation.type_gain_oracle import (
    GainEvaluationRequest,
    GainOracleRequest,
    TypeGainOracleResult,
    differentiable_type_gains,
    group_means,
    run_gain_oracle,
)
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


class TypeGainReachabilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TypeGainJacobianSummary:
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    sensitivity_cosine: float
    bounded_least_squares_residual: float
    bounded_solution_norm: float


@dataclass(frozen=True, slots=True)
class TypeGainReachabilityAudit:
    source_pair_count: int
    trial_count: int
    lag_steps: int
    type_parameter_names: tuple[str, ...]
    current_group_gains: tuple[float, float]
    teacher_group_gains: tuple[float, float]
    jacobian: tuple[tuple[float, ...], tuple[float, ...]]
    jacobian_summary: TypeGainJacobianSummary
    oracle: TypeGainOracleResult


@dataclass(frozen=True, slots=True)
class TypeGainReachabilityRequest:
    model: ResponseRetinaModel
    split: ResponseSplit
    cell_type_ids: tuple[str, ...]
    teacher_cell_gains: torch.Tensor
    lag_steps: int
    oracle_steps: int = 8


def audit_type_gain_reachability(
    request: TypeGainReachabilityRequest,
) -> TypeGainReachabilityAudit:
    model = request.model
    split = request.split
    cell_type_ids = request.cell_type_ids
    teacher_cell_gains = request.teacher_cell_gains
    lag_steps = request.lag_steps
    if teacher_cell_gains.shape != (len(cell_type_ids),):
        raise TypeGainReachabilityError("Teacher gain count must match recorded cells")
    named_type_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter_role(name) == "rgc_type_base"
    )
    if not named_type_parameters:
        raise TypeGainReachabilityError("No type-base parameters were found")
    parameters = tuple(parameter for _, parameter in named_type_parameters)
    cell_gains, group_gains = differentiable_type_gains(
        GainEvaluationRequest(model, split, cell_type_ids, lag_steps),
        create_graph=True,
    )
    rows = []
    for index in range(2):
        gradients = torch.autograd.grad(
            group_gains[index],
            parameters,
            retain_graph=index == 0,
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                tuple(
                    torch.zeros_like(parameter).flatten()
                    if gradient is None
                    else gradient.flatten()
                    for parameter, gradient in zip(parameters, gradients, strict=True)
                )
            )
        )
    jacobian = torch.stack(rows).detach().cpu().numpy().astype(np.float64)
    raw_values = torch.cat(
        tuple(parameter.detach().flatten() for parameter in parameters)
    ).cpu().numpy().astype(np.float64)
    teacher_group_gains = group_means(teacher_cell_gains, cell_type_ids)
    summary = summarize_type_gain_jacobian(
        jacobian,
        current=group_gains.detach().cpu().numpy().astype(np.float64),
        target=teacher_group_gains.detach().cpu().numpy().astype(np.float64),
        lower_delta=-8.0 - raw_values,
        upper_delta=8.0 - raw_values,
    )
    oracle = run_gain_oracle(
        GainOracleRequest(
            evaluation=GainEvaluationRequest(model, split, cell_type_ids, lag_steps),
            target_cell_gains=teacher_cell_gains.to(next(model.parameters()).device),
            named_parameters=named_type_parameters,
            max_iterations=request.oracle_steps,
        )
    )
    pairs = context_pairs(split)
    return TypeGainReachabilityAudit(
        source_pair_count=len(pairs),
        trial_count=split.spike_counts.shape[1],
        lag_steps=lag_steps,
        type_parameter_names=tuple(name for name, _ in named_type_parameters),
        current_group_gains=tuple(float(value) for value in group_gains.detach()),
        teacher_group_gains=tuple(float(value) for value in teacher_group_gains),
        jacobian=(tuple(float(value) for value in jacobian[0]), tuple(float(value) for value in jacobian[1])),
        jacobian_summary=summary,
        oracle=oracle,
    )


def summarize_type_gain_jacobian(
    jacobian: np.ndarray,
    *,
    current: np.ndarray,
    target: np.ndarray,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
) -> TypeGainJacobianSummary:
    if jacobian.ndim != 2 or jacobian.shape[0] != 2:
        raise TypeGainReachabilityError("Type-gain Jacobian must have two rows")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * singular_values[0] * np.finfo(np.float64).eps
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1]) if rank == 2 else None
    )
    denominator = np.linalg.norm(jacobian[0]) * np.linalg.norm(jacobian[1])
    cosine = 0.0 if denominator == 0 else float(jacobian[0] @ jacobian[1] / denominator)
    target_delta = target - current
    solution = bounded_least_squares(
        jacobian,
        target_delta,
        lower_delta,
        upper_delta,
    )
    return TypeGainJacobianSummary(
        rank,
        tuple(float(value) for value in singular_values),
        condition,
        cosine,
        float(np.linalg.norm(jacobian @ solution - target_delta)),
        float(np.linalg.norm(solution)),
    )


def bounded_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    solution = np.linalg.lstsq(matrix, target, rcond=None)[0]
    if np.all(solution >= lower) and np.all(solution <= upper):
        return solution
    solution = np.clip(solution, lower, upper)
    lipschitz = float(np.linalg.norm(matrix, ord=2) ** 2)
    if lipschitz == 0:
        return solution
    for _ in range(10_000):
        gradient = matrix.T @ (matrix @ solution - target)
        updated = np.clip(solution - gradient / lipschitz, lower, upper)
        if np.linalg.norm(updated - solution) <= 1e-12:
            return updated
        solution = updated
    return solution


__all__ = [
    "TypeGainJacobianSummary",
    "TypeGainOracleResult",
    "TypeGainReachabilityAudit",
    "TypeGainReachabilityError",
    "TypeGainReachabilityRequest",
    "audit_type_gain_reachability",
    "bounded_least_squares",
    "summarize_type_gain_jacobian",
]
