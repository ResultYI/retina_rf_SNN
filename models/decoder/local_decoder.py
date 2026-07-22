from __future__ import annotations

import torch
from torch import nn
class LocalDecoderError(ValueError):
    pass


class TiedLocalDecoder(nn.Module):
    def __init__(self, unit_count: int, cone_count: int, gain_max: float = 5.0) -> None:
        super().__init__()
        if unit_count < 1 or cone_count < 1:
            raise LocalDecoderError("unit_count and cone_count must be positive")
        if gain_max <= 0:
            raise LocalDecoderError("gain_max must be positive")
        initial_gain = torch.full((2, unit_count), 0.10 / gain_max)
        self.raw_unit_gain = nn.Parameter(torch.logit(initial_gain))
        self.cone_bias = nn.Parameter(torch.zeros(cone_count))
        self._unit_count = unit_count
        self._cone_count = cone_count
        self._gain_max = gain_max

    @property
    def unit_gain(self) -> torch.Tensor:
        return self._gain_max * torch.sigmoid(self.raw_unit_gain)

    @property
    def gain_max(self) -> float:
        return self._gain_max

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
