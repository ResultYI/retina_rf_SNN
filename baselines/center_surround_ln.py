from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from models.mechanistic_retina.state import decay_from_tau, fixed_one_bin_history_state


SPATIAL_SIDE: Final = 17
CONTEXT_BINS: Final = 60
POSITIVE_FLOOR: Final = 1e-6


@dataclass(frozen=True, slots=True)
class LNError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class CenterSurroundLN(nn.Module):
    def __init__(self, dt_ms: float, history_tau_ms: float, seed: int) -> None:
        super().__init__()
        if any(not math.isfinite(x) or x <= 0 for x in (dt_ms, history_tau_ms)):
            raise LNError("dt and history tau must be positive and finite")
        generator = torch.Generator().manual_seed(seed)
        self.center_xy = nn.Parameter(torch.zeros(2))
        self.raw_widths = nn.Parameter(torch.full((2,), math.log(math.expm1(1.5 - POSITIVE_FLOOR))))
        self.raw_amplitudes = nn.Parameter(torch.full((2,), math.log(math.expm1(1 - POSITIVE_FLOOR))))
        self.raw_temporal = nn.Parameter(torch.randn(2, CONTEXT_BINS, generator=generator))
        self.history_weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        axis = torch.arange(SPATIAL_SIDE, dtype=torch.float32) - (SPATIAL_SIDE - 1) / 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid_xy", torch.stack((xx, yy), dim=-1))
        self.register_buffer("history_decay", torch.tensor(decay_from_tau(dt_ms, history_tau_ms)))

    def sigmas(self) -> torch.Tensor:
        increments = F.softplus(self.raw_widths) + POSITIVE_FLOOR
        return increments.cumsum(dim=0)

    def amplitudes(self) -> torch.Tensor:
        return F.softplus(self.raw_amplitudes) + POSITIVE_FLOOR

    def gaussians(self) -> torch.Tensor:
        squared_distance = (self.grid_xy - self.center_xy).square().sum(dim=-1)
        exponents = -squared_distance[None] / (2 * self.sigmas()[:, None, None].square())
        return exponents.flatten(1).softmax(dim=1).reshape(2, SPATIAL_SIDE, SPATIAL_SIDE)

    def spatial_components(self) -> torch.Tensor:
        return self.amplitudes()[:, None, None] * self.gaussians()

    def temporal_kernels(self) -> torch.Tensor:
        norms = self.raw_temporal.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
            raise LNError("both temporal kernels must have finite nonzero norm")
        return self.raw_temporal / norms

    def pathway_kernels(self) -> torch.Tensor:
        return self.temporal_kernels()[:, :, None, None] * self.spatial_components()[:, None]

    def history_feature(self, events: torch.Tensor) -> torch.Tensor:
        return fixed_one_bin_history_state(events, float(self.history_decay))

    def forward(self, cones: torch.Tensor, observed_events: torch.Tensor) -> torch.Tensor:
        return self.forward_with_history(cones, self.history_feature(observed_events))

    def forward_with_history(self, cones: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if cones.ndim != 3 or cones.shape[-1] != SPATIAL_SIDE**2:
            raise LNError("LN stimulus must be [batch,time,289]")
        if history.shape != (*cones.shape[:2], 1):
            raise LNError("LN history must be [batch,time,1]")
        projections = (cones @ self.spatial_components().flatten(1).T).transpose(1, 2)
        kernels = self.temporal_kernels().flip(1).unsqueeze(1)
        pathways = F.conv1d(F.pad(projections, (CONTEXT_BINS - 1, 0)), kernels, groups=2)
        stimulus = (pathways[:, 0] - pathways[:, 1]).unsqueeze(-1)
        return stimulus + self.history_weight * history + self.bias

    def regularizer(self) -> torch.Tensor:
        spatial = self.spatial_components()
        differences = torch.cat((spatial.diff(dim=1).flatten(), spatial.diff(dim=2).flatten()))
        return (spatial.square().mean() + differences.square().mean()
                + self.temporal_kernels().diff(n=2, dim=1).square().mean())
