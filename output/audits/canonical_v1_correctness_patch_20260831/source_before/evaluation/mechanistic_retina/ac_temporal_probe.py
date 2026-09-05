from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch


_PROBE_NAMES: Final = ("linear_chirp", "multifrequency_flicker")


class ACTemporalProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemporalProbeConfig:
    time_steps: int = 256
    baseline_steps: int = 20
    chirp_start_hz: float = 1.0
    chirp_end_hz: float = 25.0
    flicker_hz: tuple[float, float, float] = (4.0, 11.0, 23.0)
    contrast: float = 0.8

    def __post_init__(self) -> None:
        frequencies = (
            self.chirp_start_hz,
            self.chirp_end_hz,
            *self.flicker_hz,
        )
        if self.time_steps < 32 or not 1 <= self.baseline_steps < self.time_steps - 1:
            raise ACTemporalProbeError("temporal probe has invalid duration")
        if not all(math.isfinite(value) and value > 0 for value in frequencies):
            raise ACTemporalProbeError("temporal probe frequencies must be positive")
        if self.chirp_end_hz <= self.chirp_start_hz:
            raise ACTemporalProbeError("chirp end frequency must exceed its start")
        if not math.isfinite(self.contrast) or not 0 < self.contrast <= 1:
            raise ACTemporalProbeError("temporal probe contrast must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalProbe:
    names: tuple[str, str]
    cone_response: torch.Tensor
    temporal_drive: torch.Tensor
    spatial_pattern: torch.Tensor
    time_ms: torch.Tensor
    baseline_steps: int
    dt_ms: float


def build_temporal_probe(
    cone_positions: torch.Tensor,
    dt_ms: float,
    config: TemporalProbeConfig = TemporalProbeConfig(),
) -> TemporalProbe:
    if cone_positions.ndim != 2 or cone_positions.shape[1] != 2:
        raise ACTemporalProbeError("cone positions must have shape [cone, 2]")
    if not bool(torch.isfinite(cone_positions).all()):
        raise ACTemporalProbeError("cone positions must be finite")
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ACTemporalProbeError("dt_ms must be positive and finite")

    steps = torch.arange(
        config.time_steps,
        dtype=cone_positions.dtype,
        device=cone_positions.device,
    )
    active_steps = (steps - config.baseline_steps).clamp_min(0)
    active_time_seconds = active_steps * (dt_ms / 1000.0)
    active_duration_seconds = (
        (config.time_steps - config.baseline_steps - 1) * dt_ms / 1000.0
    )
    sweep_hz_per_second = (
        config.chirp_end_hz - config.chirp_start_hz
    ) / active_duration_seconds
    chirp_phase = (
        2
        * torch.pi
        * (
            config.chirp_start_hz * active_time_seconds
            + 0.5 * sweep_hz_per_second * active_time_seconds.square()
        )
    )
    chirp = torch.sin(chirp_phase)
    flicker = (
        sum(
            weight
            * torch.sign(torch.sin(2 * torch.pi * frequency * active_time_seconds))
            for weight, frequency in zip(
                (1.0, 0.5, 0.25), config.flicker_hz, strict=True
            )
        )
        / 1.75
    )
    active = steps >= config.baseline_steps
    temporal_drive = torch.stack(
        (
            torch.where(active, chirp, torch.zeros_like(chirp)),
            torch.where(active, flicker, torch.zeros_like(flicker)),
        )
    )

    centered = cone_positions - cone_positions.mean(dim=0, keepdim=True)
    achromatic = torch.ones_like(centered[:, 0])
    opponent = torch.where(
        centered[:, 0] >= 0,
        torch.ones_like(centered[:, 0]),
        -torch.ones_like(centered[:, 0]),
    )
    spatial_pattern = torch.stack((achromatic, opponent))
    cone_response = (
        config.contrast * temporal_drive[:, :, None] * spatial_pattern[:, None, :]
    )
    return TemporalProbe(
        _PROBE_NAMES,
        cone_response,
        temporal_drive,
        spatial_pattern,
        steps * dt_ms,
        config.baseline_steps,
        dt_ms,
    )


__all__ = [
    "ACTemporalProbeError",
    "TemporalProbe",
    "TemporalProbeConfig",
    "build_temporal_probe",
]
