from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.func import functional_call

from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


@dataclass(frozen=True, slots=True)
class SubspaceOverlapRequest:
    model: MechanisticGraphTemporalRetina
    cones: torch.Tensor
    observed_counts: torch.Tensor
    tail_steps: int = 16


@dataclass(frozen=True, slots=True)
class PairwiseOverlap:
    first: str
    second: str
    maximum_canonical_correlation: float
    principal_angles_deg: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SubspaceOverlapResult:
    pairs: tuple[PairwiseOverlap, ...]
    h1_unique_fraction: float
    bc_unique_fraction: float
    ac_unique_fraction: float
    output_count: int


def fisher_subspace_overlap(request: SubspaceOverlapRequest) -> SubspaceOverlapResult:
    logits = request.model.forward_sequence(
        request.cones,
        observed_counts=request.observed_counts,
    ).logits[:, -request.tail_steps :]
    weight = torch.sigmoid(logits.detach()) * (1 - torch.sigmoid(logits.detach()))
    jacobians = {
        "H1": _jacobian(request, "gates.raw_h1_amplitude"),
        "BC": _jacobian(request, "bipolar.raw_weights"),
        "AC": torch.cat(
            tuple(
                _jacobian(request, name)
                for name in (
                    "gates.ac_local", "gates.ac_transient",
                    "amacrine.raw_tau", "amacrine.raw_delay",
                )
            ),
            dim=1,
        ),
    }
    weighted = {
        name: weight.flatten().sqrt()[:, None] * value
        for name, value in jacobians.items()
    }
    h1_q = _orthonormal(weighted["H1"])
    bc_q = _orthonormal(weighted["BC"])
    ac_q = _orthonormal(weighted["AC"])
    pairs = (
        _pair("H1", h1_q, "BC", bc_q),
        _pair("H1", h1_q, "AC", ac_q),
        _pair("BC", bc_q, "AC", ac_q),
    )
    return SubspaceOverlapResult(
        pairs,
        _unique_fraction(weighted["H1"], torch.cat((bc_q, ac_q), dim=1)),
        _unique_fraction(weighted["BC"], torch.cat((h1_q, ac_q), dim=1)),
        _unique_fraction(weighted["AC"], torch.cat((h1_q, bc_q), dim=1)),
        logits.numel(),
    )


def _jacobian(request: SubspaceOverlapRequest, state_name: str) -> torch.Tensor:
    state = dict(request.model.named_parameters()) | dict(request.model.named_buffers())
    value = state[state_name].detach().requires_grad_(True)

    def flattened(candidate: torch.Tensor) -> torch.Tensor:
        output = functional_call(
            request.model,
            {state_name: candidate},
            (request.cones,),
            {"observed_counts": request.observed_counts},
            strict=False,
        )
        return output.logits[:, -request.tail_steps :].reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        flattened,
        value,
        vectorize=True,
    )
    return jacobian.reshape(jacobian.shape[0], -1).double()


def _orthonormal(matrix: torch.Tensor) -> torch.Tensor:
    if float(matrix.norm()) == 0.0:
        return matrix.new_zeros((matrix.shape[0], 0))
    left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = singular[0] * max(matrix.shape) * torch.finfo(matrix.dtype).eps
    return left[:, singular > tolerance]


def _pair(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> PairwiseOverlap:
    if first.shape[1] == 0 or second.shape[1] == 0:
        return PairwiseOverlap(first_name, second_name, 0.0, (90.0,))
    correlations = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    angles = tuple(float(torch.rad2deg(torch.acos(value))) for value in correlations)
    return PairwiseOverlap(first_name, second_name, float(correlations.max()), angles)


def _unique_fraction(matrix: torch.Tensor, competing: torch.Tensor) -> float:
    denominator = float(matrix.square().sum())
    if denominator == 0.0:
        return 0.0
    basis = _orthonormal(competing)
    residual = matrix if basis.shape[1] == 0 else matrix - basis @ (basis.T @ matrix)
    value = float(residual.square().sum()) / denominator
    return min(1.0, max(0.0, value))


__all__ = [
    "PairwiseOverlap",
    "SubspaceOverlapRequest",
    "SubspaceOverlapResult",
    "fisher_subspace_overlap",
]
