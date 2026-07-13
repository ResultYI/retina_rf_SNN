from __future__ import annotations

import torch
from torch import nn


def raw_ordered_taus(
    initial_ms: torch.Tensor,
    bounds_ms: torch.Tensor,
) -> tuple[nn.Parameter, nn.Parameter]:
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
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    transient_max = torch.minimum(transient_bounds[1], sustained_bounds[1])
    transient = transient_bounds[0] + (
        transient_max - transient_bounds[0]
    ) * torch.sigmoid(raw_transient)
    ordering_epsilon = torch.finfo(bounds_ms.dtype).eps * sustained_bounds[1]
    sustained_min = torch.maximum(
        sustained_bounds[0],
        transient + ordering_epsilon,
    )
    sustained = sustained_min + (
        sustained_bounds[1] - sustained_min
    ) * torch.sigmoid(raw_sustained)
    return torch.stack((sustained, transient))
