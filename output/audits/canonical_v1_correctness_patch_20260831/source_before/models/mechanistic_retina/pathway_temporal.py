from __future__ import annotations

import torch

from models.mechanistic_retina.state import causal_fractional_delay


def pathway_temporal_basis(
    tau_ms: torch.Tensor,
    explicit_delay_ms: torch.Tensor,
    lag_steps: int,
    dt_ms: float,
) -> torch.Tensor:
    elapsed = (
        torch.arange(
            lag_steps - 1,
            -1,
            -1,
            dtype=tau_ms.dtype,
            device=tau_ms.device,
        )
        * dt_ms
    )
    scaled = elapsed.view(1, 1, -1) / tau_ms[..., None]
    modes = scaled * torch.exp(1 - scaled)
    normalized = modes / modes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    semantic_lag = normalized.flip(-1).permute(2, 0, 1).reshape(1, lag_steps, -1)
    channel_delays = explicit_delay_ms[:, None].expand_as(tau_ms).reshape(-1)
    delayed = causal_fractional_delay(semantic_lag, channel_delays, dt_ms=dt_ms)
    return delayed.reshape(lag_steps, *tau_ms.shape).permute(1, 2, 0).flip(-1)


__all__ = ["pathway_temporal_basis"]
