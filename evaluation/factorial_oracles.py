from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.factorial_contrasts import FactorialContrasts, factorial_contrasts
from evaluation.parameter_audit import ParameterRole, parameter_role
from evaluation.type_gain_oracle import (
    GainEvaluationRequest,
    GainOracleRequest,
    TypeGainOracleResult,
    differentiable_type_gains,
    run_gain_oracle,
)


class FactorialOracleError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FactorialOracleRequest:
    evaluation: GainEvaluationRequest
    target_cell_gains: torch.Tensor
    max_iterations: int = 40
    tolerance: float = 1e-3


@dataclass(frozen=True, slots=True)
class FactorialOracleCase:
    label: str
    roles: tuple[ParameterRole, ...]
    parameter_names: tuple[str, ...]
    target_cell_gains: tuple[float, ...]
    target_contrasts: FactorialContrasts
    result: TypeGainOracleResult
    after_contrasts: FactorialContrasts
    contrast_errors: FactorialContrasts
    passed: bool


@dataclass(frozen=True, slots=True)
class FactorialOracleAudit:
    current_cell_gains: tuple[float, ...]
    current_contrasts: FactorialContrasts
    type_only_target_cell_gains: tuple[float, ...]
    factorial_target_cell_gains: tuple[float, ...]
    cases: tuple[FactorialOracleCase, ...]


@dataclass(frozen=True, slots=True)
class _CaseRequest:
    request: FactorialOracleRequest
    label: str
    roles: tuple[ParameterRole, ...]
    target: torch.Tensor


def audit_factorial_oracles(request: FactorialOracleRequest) -> FactorialOracleAudit:
    _validate_request(request)
    current, _ = differentiable_type_gains(request.evaluation, create_graph=False)
    target = request.target_cell_gains.to(current.device)
    type_only = torch.stack(
        (target[:2].mean(), target[:2].mean(), target[2:].mean(), target[2:].mean())
    )
    cases = tuple(
        _run_case(case)
        for case in (
            _CaseRequest(request, "A_type_only_target", ("rgc_type_base",), type_only),
            _CaseRequest(
                request,
                "B_factorial_type_plus_polarity",
                ("rgc_type_base", "polarity_pathway"),
                target,
            ),
            _CaseRequest(
                request,
                "C_factorial_with_residual",
                ("rgc_type_base", "polarity_pathway", "rgc_cell_residual"),
                target,
            ),
        )
    )
    return FactorialOracleAudit(
        tuple(float(value) for value in current),
        factorial_contrasts(current.detach().cpu().numpy()),
        tuple(float(value) for value in type_only),
        tuple(float(value) for value in target),
        cases,
    )


def _validate_request(request: FactorialOracleRequest) -> None:
    if request.evaluation.cell_type_ids != ("midget", "midget", "parasol", "parasol"):
        raise FactorialOracleError("Factorial oracle requires the canonical four-cell order")
    if request.target_cell_gains.shape != (4,):
        raise FactorialOracleError("Factorial oracle target must contain four gains")


def _run_case(case: _CaseRequest) -> FactorialOracleCase:
    model = case.request.evaluation.model
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter_role(name) in case.roles
    )
    if not named:
        raise FactorialOracleError(f"No parameters found for {case.label}")
    result = run_gain_oracle(
        GainOracleRequest(
            evaluation=case.request.evaluation,
            target_cell_gains=case.target,
            named_parameters=named,
            max_iterations=case.request.max_iterations,
            tolerance=case.request.tolerance,
        )
    )
    target_values = case.target.detach().cpu().numpy().astype(np.float64)
    after_values = np.asarray(result.after_cell_gains, dtype=np.float64)
    target_contrasts = factorial_contrasts(target_values)
    after_contrasts = factorial_contrasts(after_values)
    return FactorialOracleCase(
        case.label,
        case.roles,
        tuple(name for name, _ in named),
        tuple(float(value) for value in target_values),
        target_contrasts,
        result,
        after_contrasts,
        factorial_contrasts(after_values - target_values),
        result.after_direction_count == 4 and result.converged,
    )


__all__ = [
    "FactorialOracleAudit",
    "FactorialOracleCase",
    "FactorialOracleError",
    "FactorialOracleRequest",
    "audit_factorial_oracles",
]
