from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalDecoderError(ValueError):
    pass


class TiedLocalDecoder(nn.Module):
    def __init__(self, unit_count: int, cone_count: int) -> None:
        super().__init__()
        if unit_count < 1 or cone_count < 1:
            raise LocalDecoderError("unit_count and cone_count must be positive")
        self.raw_unit_gain = nn.Parameter(torch.full((2, unit_count), -2.0))
        self.cone_bias = nn.Parameter(torch.zeros(cone_count))
        self._unit_count = unit_count
        self._cone_count = cone_count

    @property
    def unit_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_unit_gain)

    def forward(self, rates: torch.Tensor, spatial_weights: torch.Tensor) -> torch.Tensor:
        if rates.ndim != 4 or rates.shape[-2:] != (2, self._unit_count):
            raise LocalDecoderError("rates must have shape [batch,time,polarity,unit]")
        if spatial_weights.shape != (self._unit_count, self._cone_count):
            raise LocalDecoderError("spatial_weights must have shape [unit,cone]")
        signed = (
            self.unit_gain[0] * rates[:, :, 0]
            - self.unit_gain[1] * rates[:, :, 1]
        )
        return signed @ spatial_weights + self.cone_bias


__all__ = ["LocalDecoderError", "TiedLocalDecoder"]

