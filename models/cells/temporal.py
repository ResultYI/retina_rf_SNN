from __future__ import annotations

import torch
from torch import nn


class OrderedTauError(ValueError):
    pass


def raw_ordered_taus(
    initial_ms: torch.Tensor,
    bounds_ms: torch.Tensor,
) -> tuple[nn.Parameter, nn.Parameter]:
    _validate_bounds(bounds_ms)
    sustained_initial, transient_initial = initial_ms.unbind()
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    transient_max = torch.minimum(transient_bounds[1], sustained_bounds[1])
    transient_fraction = (transient_initial - transient_bounds[0]) / (
        transient_max - transient_bounds[0]
    )
    ordering_epsilon = torch.finfo(bounds_ms.dtype).eps * sustained_bounds[1]
    sustained_min = torch.maximum(
        sustained_bounds[0],
        transient_initial + ordering_epsilon,
    )
    sustained_fraction = (sustained_initial - sustained_min) / (
        sustained_bounds[1] - sustained_min
    )
    return (
        nn.Parameter(torch.logit(sustained_fraction)),
        nn.Parameter(torch.logit(transient_fraction)),
    )


def ordered_taus(
    raw_sustained: torch.Tensor,
    raw_transient: torch.Tensor,
    bounds_ms: torch.Tensor,
) -> torch.Tensor:
    ordering_epsilon = _validate_bounds(bounds_ms)
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    transient_max = torch.minimum(transient_bounds[1], sustained_bounds[1])
    transient = transient_bounds[0] + (
        transient_max - transient_bounds[0]
    ) * torch.sigmoid(raw_transient)
    sustained_min = torch.maximum(
        sustained_bounds[0],
        transient + ordering_epsilon,
    )
    sustained = sustained_min + (
        sustained_bounds[1] - sustained_min
    ) * torch.sigmoid(raw_sustained)
    if not bool((transient < sustained).all()):
        raise OrderedTauError("transient tau must remain below sustained tau")
    if not bool((transient <= sustained_bounds[1] - ordering_epsilon).all()):
        raise OrderedTauError("transient tau exceeds the ordered sustained bound")
    return torch.stack((sustained, transient))


def _validate_bounds(bounds_ms: torch.Tensor) -> torch.Tensor:
    if bounds_ms.shape != (2, 2) or not bool(torch.isfinite(bounds_ms).all()):
        raise OrderedTauError("tau bounds must be finite [sustained/transient, lower/upper]")
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    epsilon = torch.finfo(bounds_ms.dtype).eps * sustained_bounds[1]
    if not bool(epsilon > 0):
        raise OrderedTauError("tau ordering epsilon must be positive")
    if not bool(sustained_bounds[0] < sustained_bounds[1]):
        raise OrderedTauError("sustained tau lower bound must be below its upper bound")
    effective_transient_upper = torch.minimum(
        transient_bounds[1],
        sustained_bounds[1] - epsilon,
    )
    if not bool(transient_bounds[0] < effective_transient_upper):
        raise OrderedTauError(
            "transient tau lower bound must fit below both transient and ordered sustained upper bounds"
        )
    return epsilon


__all__ = ["OrderedTauError", "ordered_taus", "raw_ordered_taus"]
