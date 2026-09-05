from __future__ import annotations

import torch
from torch import nn


class PathwayLocalOperator(nn.Module):
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        channel_count = 4 * 2 * 3
        self.depthwise = nn.Conv1d(
            channel_count,
            channel_count,
            kernel_size=3,
            padding=2,
            groups=channel_count,
            bias=True,
        )
        nn.init.zeros_(self.depthwise.weight)
        nn.init.zeros_(self.depthwise.bias)
        self.depthwise.requires_grad_(False)
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, features: torch.Tensor, *, enabled: bool) -> torch.Tensor:
        target_shape = features.shape
        if not enabled:
            return torch.ones(target_shape, dtype=features.dtype, device=features.device)
        batch, time, cells, paths, spatial, temporal = target_shape
        signal = features.permute(0, 2, 3, 4, 5, 1).reshape(
            batch * cells, paths * spatial * temporal, time
        )
        raw = self.depthwise(signal)[..., :time]
        bounded = torch.exp(self.epsilon * torch.tanh(raw))
        return bounded.reshape(batch, cells, paths, spatial, temporal, time).permute(
            0, 5, 1, 2, 3, 4
        )


__all__ = ["PathwayLocalOperator"]
