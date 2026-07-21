from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def raw_parameter(initial: float, minimum: float, maximum: float) -> nn.Parameter:
    return nn.Parameter(raw_value(initial, minimum, maximum))


def raw_pair_parameter(
    first: float,
    second: float,
    minimum: float,
    maximum: float,
) -> nn.Parameter:
    return nn.Parameter(
        torch.stack(
            (
                raw_value(first, minimum, maximum),
                raw_value(second, minimum, maximum),
            )
        )
    )


def raw_value(initial: float, minimum: float, maximum: float) -> torch.Tensor:
    fraction = (initial - minimum) / (maximum - minimum)
    return torch.logit(torch.tensor(fraction))


def bounded(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    minimum, maximum = bounds
    return minimum + (maximum - minimum) * torch.sigmoid(raw)


def smooth_rectify(value: torch.Tensor, softness: torch.Tensor) -> torch.Tensor:
    return softness * F.softplus(value / softness)
