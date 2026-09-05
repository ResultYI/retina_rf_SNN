from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from baselines.center_surround_ln import CONTEXT_BINS, SPATIAL_SIDE, LNError
from models.mechanistic_retina.state import decay_from_tau, fixed_one_bin_history_state


READOUT_SIDE: Final = SPATIAL_SIDE - (5 - 1) - (3 - 1)


class CompactCausalCNN(nn.Module):
    def __init__(self, dt_ms: float, history_tau_ms: float, seed: int) -> None:
        super().__init__()
        if any(not math.isfinite(x) or x <= 0 for x in (dt_ms, history_tau_ms)):
            raise LNError("dt and history tau must be positive and finite")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.conv1 = nn.Conv3d(1, 4, (12, 5, 5))
            self.conv2 = nn.Conv3d(4, 4, (9, 3, 3), dilation=(6, 1, 1))
            self.spatial_readout = nn.Parameter(torch.empty(4, READOUT_SIDE, READOUT_SIDE))
            nn.init.uniform_(self.spatial_readout, -1 / math.sqrt(self.spatial_readout.numel()),
                             1 / math.sqrt(self.spatial_readout.numel()))
        self.history_weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("history_decay", torch.tensor(decay_from_tau(dt_ms, history_tau_ms)))

    def history_feature(self, events: torch.Tensor) -> torch.Tensor:
        return fixed_one_bin_history_state(events, float(self.history_decay))

    def forward(self, cones: torch.Tensor, observed_events: torch.Tensor) -> torch.Tensor:
        return self.forward_with_history(cones, self.history_feature(observed_events))

    def forward_with_history(self, cones: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if cones.ndim != 3 or cones.shape[-1] != SPATIAL_SIDE**2:
            raise LNError("CNN stimulus must be [batch,time,289]")
        if history.shape != (*cones.shape[:2], 1):
            raise LNError("CNN history must be [batch,time,1]")
        batch, time, _ = cones.shape
        stimulus = cones.reshape(batch, 1, time, SPATIAL_SIDE, SPATIAL_SIDE)
        first = F.relu(self.conv1(F.pad(stimulus, (0, 0, 0, 0, 11, 0))))
        second = F.relu(self.conv2(F.pad(first, (0, 0, 0, 0, CONTEXT_BINS - 12, 0))))
        activation = torch.einsum("bcthw,chw->bt", second, self.spatial_readout).unsqueeze(-1)
        return activation + self.history_weight * history + self.bias
