from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class MechanisticStateError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def decay_from_tau(dt_ms: float, tau_ms: float) -> float:
    return float(torch.exp(torch.tensor(-dt_ms / tau_ms)))


def causal_lowpass(values: torch.Tensor, decay: float | torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise MechanisticStateError(
            "causal low-pass input must be [batch,time,channel]"
        )
    channel_count = values.shape[-1]
    alpha = torch.as_tensor(decay, dtype=values.dtype, device=values.device).reshape(-1)
    if alpha.numel() == 1:
        alpha = alpha.expand(channel_count)
    if alpha.numel() != channel_count:
        raise MechanisticStateError(
            "causal low-pass decay count differs from channel count"
        )
    time_count = values.shape[1]
    ages = torch.arange(
        time_count - 1,
        -1,
        -1,
        dtype=values.dtype,
        device=values.device,
    )
    kernels = (1 - alpha[:, None]) * alpha[:, None].pow(ages[None, :])
    channels_first = values.transpose(1, 2)
    filtered = F.conv1d(
        F.pad(channels_first, (time_count - 1, 0)),
        kernels[:, None, :],
        groups=channel_count,
    )
    return filtered.transpose(1, 2)


def causal_fractional_delay(
    values: torch.Tensor,
    delay_ms: float | torch.Tensor,
    *,
    dt_ms: float,
) -> torch.Tensor:
    if values.ndim != 3:
        raise MechanisticStateError(
            "fractional-delay input must be [batch,time,channel]"
        )
    if not torch.isfinite(torch.tensor(dt_ms)) or dt_ms <= 0:
        raise MechanisticStateError(
            "fractional-delay dt_ms must be positive and finite"
        )
    channel_count = values.shape[-1]
    delays = torch.as_tensor(
        delay_ms, dtype=values.dtype, device=values.device
    ).reshape(-1)
    if delays.numel() == 1:
        delays = delays.expand(channel_count)
    if delays.numel() != channel_count:
        raise MechanisticStateError("fractional-delay count differs from channel count")
    if not bool(torch.isfinite(delays).all()) or bool(torch.any(delays < 0)):
        raise MechanisticStateError("fractional delays must be finite and non-negative")

    delay_steps = delays / dt_ms
    lower_steps = torch.floor(delay_steps)
    fraction = delay_steps - lower_steps
    time_index = torch.arange(values.shape[1], device=values.device).view(-1, 1)
    recent_index = time_index - lower_steps.to(torch.long).view(1, -1)
    older_index = recent_index - 1

    def sample(index: torch.Tensor) -> torch.Tensor:
        valid = index >= 0
        safe = index.clamp(0, values.shape[1] - 1)
        gathered = values.gather(1, safe.unsqueeze(0).expand(values.shape[0], -1, -1))
        return gathered * valid.to(values.dtype).unsqueeze(0)

    recent = sample(recent_index)
    older = sample(older_index)
    weight = fraction.view(1, 1, -1)
    return recent * (1 - weight) + older * weight


def fixed_one_bin_history_state(events: torch.Tensor, decay: float) -> torch.Tensor:
    shifted = torch.cat((torch.zeros_like(events[:, :1]), events[:, :-1]), dim=1)
    return causal_lowpass(shifted, decay)


__all__ = [
    "MechanisticStateError",
    "causal_fractional_delay",
    "causal_lowpass",
    "decay_from_tau",
    "fixed_one_bin_history_state",
]
