from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.factorial_contrasts import FactorialContrasts, factorial_contrasts
from evaluation.parameter_audit import ParameterRole, parameter_role
from evaluation.type_gain_oracle import GainEvaluationRequest, differentiable_type_gains
from evaluation.type_gain_reachability import bounded_least_squares
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


class FactorialReachabilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FactorialJacobianInput:
    jacobian: np.ndarray
    current: np.ndarray
    target: np.ndarray
    lower_delta: np.ndarray
    upper_delta: np.ndarray


@dataclass(frozen=True, slots=True)
class FactorialJacobianSummary:
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    bounded_least_squares_residual: float
    bounded_solution_norm: float
    predicted_cell_gains: tuple[float, ...]
    predicted_contrasts: FactorialContrasts


@dataclass(frozen=True, slots=True)
class FactorialParameterAudit:
    label: str
    roles: tuple[ParameterRole, ...]
    parameter_names: tuple[str, ...]
    jacobian: tuple[tuple[float, ...], ...]
    summary: FactorialJacobianSummary


@dataclass(frozen=True, slots=True)
class FactorialReachabilityRequest:
    model: ResponseRetinaModel
    split: ResponseSplit
    cell_type_ids: tuple[str, ...]
    polarities: tuple[int, ...]
    target_cell_gains: torch.Tensor
    lag_steps: int


@dataclass(frozen=True, slots=True)
class FactorialReachabilityAudit:
    current_cell_gains: tuple[float, ...]
    target_cell_gains: tuple[float, ...]
    current_contrasts: FactorialContrasts
    target_contrasts: FactorialContrasts
    parameter_audits: tuple[FactorialParameterAudit, ...]


@dataclass(frozen=True, slots=True)
class _GroupAuditInput:
    label: str
    roles: tuple[ParameterRole, ...]
    named: tuple[tuple[str, torch.nn.Parameter], ...]
    blocks: dict[str, tuple[np.ndarray, ...]]
    current: np.ndarray
    target: np.ndarray


_GROUPS: tuple[tuple[str, tuple[ParameterRole, ...]], ...] = (
    ("A_type_base", ("rgc_type_base",)),
    ("B_polarity_pathway", ("polarity_pathway",)),
    ("C_type_plus_polarity", ("rgc_type_base", "polarity_pathway")),
    (
        "D_type_plus_polarity_plus_residual",
        ("rgc_type_base", "polarity_pathway", "rgc_cell_residual"),
    ),
)


def audit_factorial_reachability(
    request: FactorialReachabilityRequest,
) -> FactorialReachabilityAudit:
    _validate_layout(request)
    evaluation = GainEvaluationRequest(
        request.model,
        request.split,
        request.cell_type_ids,
        request.lag_steps,
    )
    cell_gains, _ = differentiable_type_gains(evaluation, create_graph=True)
    named = tuple(
        (name, parameter)
        for name, parameter in request.model.named_parameters()
        if parameter_role(name) in {role for _, roles in _GROUPS for role in roles}
    )
    blocks = _cell_jacobian_blocks(cell_gains, named)
    current = cell_gains.detach().cpu().numpy().astype(np.float64)
    target = request.target_cell_gains.detach().cpu().numpy().astype(np.float64)
    audits = tuple(
        _group_audit(_GroupAuditInput(label, roles, named, blocks, current, target))
        for label, roles in _GROUPS
    )
    return FactorialReachabilityAudit(
        tuple(float(value) for value in current),
        tuple(float(value) for value in target),
        factorial_contrasts(current),
        factorial_contrasts(target),
        audits,
    )


def summarize_factorial_jacobian(
    values: FactorialJacobianInput,
) -> FactorialJacobianSummary:
    matrix = np.asarray(values.jacobian, dtype=np.float64)
    current = np.asarray(values.current, dtype=np.float64)
    target = np.asarray(values.target, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 4 or current.shape != (4,) or target.shape != (4,):
        raise FactorialReachabilityError("Factorial Jacobian must predict four cells")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * singular_values[0] * np.finfo(np.float64).eps
    rank = int(np.sum(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if rank == 4 else None
    solution = bounded_least_squares(
        matrix,
        target - current,
        np.asarray(values.lower_delta, dtype=np.float64),
        np.asarray(values.upper_delta, dtype=np.float64),
    )
    predicted = current + matrix @ solution
    return FactorialJacobianSummary(
        rank,
        tuple(float(value) for value in singular_values),
        condition,
        float(np.linalg.norm(predicted - target)),
        float(np.linalg.norm(solution)),
        tuple(float(value) for value in predicted),
        factorial_contrasts(predicted),
    )


def _validate_layout(request: FactorialReachabilityRequest) -> None:
    expected_types = ("midget", "midget", "parasol", "parasol")
    if request.cell_type_ids != expected_types or request.polarities != (0, 1, 0, 1):
        raise FactorialReachabilityError(
            "Factorial audit requires midget ON/OFF then parasol ON/OFF cell order"
        )
    if request.target_cell_gains.shape != (4,):
        raise FactorialReachabilityError("Factorial target must contain four gains")


def _cell_jacobian_blocks(
    cell_gains: torch.Tensor,
    named: tuple[tuple[str, torch.nn.Parameter], ...],
) -> dict[str, tuple[np.ndarray, ...]]:
    parameters = tuple(parameter for _, parameter in named)
    by_name: dict[str, list[np.ndarray]] = {name: [] for name, _ in named}
    for cell in range(4):
        gradients = torch.autograd.grad(
            cell_gains[cell],
            parameters,
            retain_graph=cell < 3,
            allow_unused=True,
        )
        for (name, parameter), gradient in zip(named, gradients, strict=True):
            value = torch.zeros_like(parameter) if gradient is None else gradient
            by_name[name].append(value.detach().flatten().cpu().numpy().astype(np.float64))
    return {name: tuple(values) for name, values in by_name.items()}


def _group_audit(values: _GroupAuditInput) -> FactorialParameterAudit:
    selected = tuple(
        (name, parameter)
        for name, parameter in values.named
        if parameter_role(name) in values.roles
    )
    if not selected:
        raise FactorialReachabilityError(f"No parameters found for {values.label}")
    matrix = np.stack(
        [
            np.concatenate([values.blocks[name][cell] for name, _ in selected])
            for cell in range(4)
        ]
    )
    raw = np.concatenate(
        [parameter.detach().flatten().cpu().numpy().astype(np.float64) for _, parameter in selected]
    )
    summary = summarize_factorial_jacobian(
        FactorialJacobianInput(
            matrix,
            values.current,
            values.target,
            -8.0 - raw,
            8.0 - raw,
        )
    )
    return FactorialParameterAudit(
        values.label,
        values.roles,
        tuple(name for name, _ in selected),
        tuple(tuple(float(value) for value in row) for row in matrix),
        summary,
    )


__all__ = [
    "FactorialJacobianInput",
    "FactorialJacobianSummary",
    "FactorialParameterAudit",
    "FactorialReachabilityAudit",
    "FactorialReachabilityError",
    "FactorialReachabilityRequest",
    "audit_factorial_reachability",
    "summarize_factorial_jacobian",
]
