from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch


_PATH_SIGNS: Final = (1.0, 1.0, -1.0, -1.0)
_BASIS_PER_PATH: Final = 6
_KKT_TOLERANCE: Final = 1e-10


@dataclass(frozen=True, slots=True)
class NonnegativeProjectionCell:
    iterations: int
    kkt_residual: float
    excluded_energy: float
    boundary_fraction: float
    converged: bool


@dataclass(frozen=True, slots=True)
class NonnegativeBasisProjection:
    weights: torch.Tensor
    projected: torch.Tensor
    cells: tuple[NonnegativeProjectionCell, ...]


@dataclass(frozen=True, slots=True)
class _NNLSSolution:
    values: np.ndarray
    iterations: int
    kkt_residual: float
    converged: bool


@dataclass(frozen=True, slots=True)
class CandidateProjectionError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def physiological_path_signs(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(_PATH_SIGNS, dtype=dtype, device=device).repeat_interleave(
        _BASIS_PER_PATH
    )


def project_rf_columns_nonnegative(
    columns: torch.Tensor,
    target: torch.Tensor,
) -> NonnegativeBasisProjection:
    if columns.ndim != 4 or target.shape != columns.shape[:3]:
        raise CandidateProjectionError(
            "nonnegative projection requires [context,cell,feature,basis] columns"
        )
    if columns.shape[-1] != len(_PATH_SIGNS) * _BASIS_PER_PATH:
        raise CandidateProjectionError(
            "nonnegative projection requires 24 V4 basis columns"
        )
    if not bool(torch.isfinite(columns).all() and torch.isfinite(target).all()):
        raise CandidateProjectionError("projection inputs must be finite")
    context_count, cell_count, feature_count, _ = columns.shape
    signs = physiological_path_signs().numpy()
    weights = []
    projected = []
    diagnostics = []
    for cell in range(cell_count):
        matrix = columns[:, cell].detach().reshape(-1, 24).double().cpu().numpy()
        signed_matrix = matrix * signs
        teacher = target[:, cell].detach().reshape(-1).double().cpu().numpy()
        solution = _solve_nnls(signed_matrix, teacher)
        fitted = signed_matrix @ solution.values
        denominator = max(float(teacher @ teacher), np.finfo(np.float64).tiny)
        diagnostics.append(
            NonnegativeProjectionCell(
                solution.iterations,
                solution.kkt_residual,
                float(np.square(teacher - fitted).sum() / denominator),
                float(np.mean(solution.values <= _KKT_TOLERANCE)),
                solution.converged,
            )
        )
        weights.append(torch.from_numpy(solution.values.copy()))
        projected.append(
            torch.from_numpy(fitted.copy()).reshape(context_count, feature_count)
        )
    return NonnegativeBasisProjection(
        torch.stack(weights),
        torch.stack(projected, dim=1),
        tuple(diagnostics),
    )


def _solve_nnls(matrix: np.ndarray, target: np.ndarray) -> _NNLSSolution:
    variable_count = matrix.shape[1]
    values = np.zeros(variable_count, dtype=np.float64)
    passive = np.zeros(variable_count, dtype=bool)
    scale = max(1.0, float(np.linalg.norm(matrix.T @ target, ord=np.inf)))
    maximum_iterations = 10 * variable_count * variable_count
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        dual = matrix.T @ (target - matrix @ values)
        eligible = np.where(passive, -np.inf, dual)
        if float(np.max(eligible)) <= _KKT_TOLERANCE * scale:
            residual = _kkt_residual(matrix, target, values)
            return _NNLSSolution(values, iterations, residual, True)
        passive[int(np.argmax(eligible))] = True
        while True:
            candidate = np.zeros_like(values)
            candidate[passive] = np.linalg.lstsq(
                matrix[:, passive], target, rcond=None
            )[0]
            if bool(np.all(candidate[passive] > _KKT_TOLERANCE)):
                values = candidate
                break
            blocked = passive & (candidate <= _KKT_TOLERANCE)
            ratios = values[blocked] / (values[blocked] - candidate[blocked])
            alpha = float(np.min(ratios)) if ratios.size else 0.0
            values += alpha * (candidate - values)
            removable = passive & (values <= _KKT_TOLERANCE)
            values[removable] = 0.0
            passive[removable] = False
    residual = _kkt_residual(matrix, target, values)
    return _NNLSSolution(
        values, iterations, residual, residual <= _KKT_TOLERANCE * scale
    )


def _kkt_residual(
    matrix: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
) -> float:
    gradient = matrix.T @ (matrix @ values - target)
    active = values > _KKT_TOLERANCE
    active_residual = float(np.max(np.abs(gradient[active]))) if np.any(active) else 0.0
    boundary_residual = (
        float(np.max(np.maximum(-gradient[~active], 0.0))) if np.any(~active) else 0.0
    )
    return max(active_residual, boundary_residual)


__all__ = [
    "CandidateProjectionError",
    "NonnegativeBasisProjection",
    "NonnegativeProjectionCell",
    "physiological_path_signs",
    "project_rf_columns_nonnegative",
]
