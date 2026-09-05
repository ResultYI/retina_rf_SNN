from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch


TEMPORAL_OFFSETS_MS: Final = (-100.0, -50.0, 0.0, 50.0, 100.0)
_COMBINED_NAMES: Final = (
    "surround_then_center_100ms",
    "surround_then_center_50ms",
    "center_surround_simultaneous",
    "center_then_surround_50ms",
    "center_then_surround_100ms",
)


@dataclass(frozen=True, slots=True)
class CenterSurroundProbeError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CenterSurroundProbeConfig:
    total_ms: float = 1000.0
    center_onset_ms: float = 300.0
    pulse_duration_ms: float = 100.0
    contrast: float = 0.25
    event_window_ms: float = 50.0

    def __post_init__(self) -> None:
        values = (
            self.total_ms,
            self.center_onset_ms,
            self.pulse_duration_ms,
            self.contrast,
            self.event_window_ms,
        )
        earliest = self.center_onset_ms + min(TEMPORAL_OFFSETS_MS)
        latest = (
            self.center_onset_ms
            + max(TEMPORAL_OFFSETS_MS)
            + self.pulse_duration_ms
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise CenterSurroundProbeError("probe values must be positive and finite")
        if earliest < 0 or latest > self.total_ms:
            raise CenterSurroundProbeError("probe events must lie inside the time axis")
        if self.contrast > 1:
            raise CenterSurroundProbeError("probe contrast must not exceed one")


@dataclass(frozen=True, slots=True)
class CenterSurroundProbe:
    names: tuple[str, ...]
    cone_drive: torch.Tensor
    time_ms: torch.Tensor
    offset_ms: torch.Tensor
    center_onset_ms: torch.Tensor
    surround_onset_ms: torch.Tensor
    dt_ms: float
    pulse_duration_ms: float


@dataclass(frozen=True, slots=True)
class ResponseSummary:
    peak_response: float
    peak_absolute_response: float
    peak_latency_ms: float
    response_integral: float
    center_onset_response: float | None
    center_offset_response: float | None
    surround_onset_response: float | None
    surround_offset_response: float | None


def build_center_surround_probe(
    center_support: torch.Tensor,
    surround_support: torch.Tensor,
    dt_ms: float,
    polarity_sign: float,
    config: CenterSurroundProbeConfig = CenterSurroundProbeConfig(),
) -> CenterSurroundProbe:
    if (
        center_support.ndim != 1
        or surround_support.shape != center_support.shape
        or not bool(torch.isfinite(center_support).all())
        or not bool(torch.isfinite(surround_support).all())
    ):
        raise CenterSurroundProbeError("supports must be finite vectors of equal shape")
    if bool(((center_support > 0) & (surround_support > 0)).any()):
        raise CenterSurroundProbeError("center and surround supports must be disjoint")
    if not bool((center_support > 0).any() and (surround_support > 0).any()):
        raise CenterSurroundProbeError("center and surround supports must be nonempty")
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise CenterSurroundProbeError("dt_ms must be positive and finite")
    if polarity_sign not in (-1.0, 1.0):
        raise CenterSurroundProbeError("polarity sign must be -1 or +1")

    steps = math.ceil(config.total_ms / dt_ms)
    time_ms = torch.arange(
        steps, dtype=center_support.dtype, device=center_support.device
    ) * dt_ms
    nan = float("nan")
    center_onsets = (config.center_onset_ms, nan) + (config.center_onset_ms,) * 5
    surround_onsets = (nan, config.center_onset_ms) + tuple(
        config.center_onset_ms + offset for offset in TEMPORAL_OFFSETS_MS
    )
    names = ("center_only", "surround_only") + _COMBINED_NAMES
    drives = []
    for center_onset, surround_onset in zip(
        center_onsets, surround_onsets, strict=True
    ):
        center_drive = _optional_box(time_ms, center_onset, config.pulse_duration_ms, dt_ms)
        surround_drive = _optional_box(
            time_ms, surround_onset, config.pulse_duration_ms, dt_ms
        )
        drives.append(
            polarity_sign
            * config.contrast
            * (
                center_drive[:, None] * (center_support > 0)
                + surround_drive[:, None] * (surround_support > 0)
            )
        )
    return CenterSurroundProbe(
        names,
        torch.stack(drives),
        time_ms,
        time_ms.new_tensor((nan, nan, *TEMPORAL_OFFSETS_MS)),
        time_ms.new_tensor(center_onsets),
        time_ms.new_tensor(surround_onsets),
        dt_ms,
        config.pulse_duration_ms,
    )


def summarize_response(
    response: torch.Tensor,
    *,
    dt_ms: float,
    center_onset_ms: float | None,
    surround_onset_ms: float | None,
    pulse_duration_ms: float,
    event_window_ms: float,
) -> ResponseSummary:
    if response.ndim != 1 or not bool(torch.isfinite(response).all()):
        raise CenterSurroundProbeError("response must be a finite time vector")
    peak_index = int(response.abs().argmax())
    return ResponseSummary(
        float(response[peak_index]),
        float(response[peak_index].abs()),
        peak_index * dt_ms,
        float(response.sum() * (dt_ms / 1000.0)),
        _event_mean(response, center_onset_ms, dt_ms, event_window_ms),
        _event_mean(
            response,
            None if center_onset_ms is None else center_onset_ms + pulse_duration_ms,
            dt_ms,
            event_window_ms,
        ),
        _event_mean(response, surround_onset_ms, dt_ms, event_window_ms),
        _event_mean(
            response,
            None if surround_onset_ms is None else surround_onset_ms + pulse_duration_ms,
            dt_ms,
            event_window_ms,
        ),
    )


def _optional_box(
    time_ms: torch.Tensor, onset_ms: float, duration_ms: float, dt_ms: float
) -> torch.Tensor:
    if math.isnan(onset_ms):
        return torch.zeros_like(time_ms)
    left = torch.maximum(time_ms, time_ms.new_tensor(onset_ms))
    right = torch.minimum(time_ms + dt_ms, time_ms.new_tensor(onset_ms + duration_ms))
    return ((right - left) / dt_ms).clamp(0, 1)


def _event_mean(
    response: torch.Tensor,
    event_ms: float | None,
    dt_ms: float,
    window_ms: float,
) -> float | None:
    if event_ms is None:
        return None
    time_ms = torch.arange(response.numel(), device=response.device) * dt_ms
    active = (time_ms >= event_ms) & (time_ms < event_ms + window_ms)
    return float(response[active].mean())


__all__ = [
    "CenterSurroundProbe",
    "CenterSurroundProbeConfig",
    "CenterSurroundProbeError",
    "ResponseSummary",
    "TEMPORAL_OFFSETS_MS",
    "build_center_surround_probe",
    "summarize_response",
]
