from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch


class TemporalEvaluationError(ValueError):
    pass


class TemporalProbeKind(StrEnum):
    IMPULSE = "impulse"
    STEP = "step"
    FLICKER = "flicker"
    CHIRP = "chirp"


@dataclass(frozen=True, slots=True)
class TemporalProbeSpec:
    cone_count: int
    time_steps: int
    dt_ms: float
    onset_step: int
    offset_step: int
    amplitude: float
    flicker_hz: float
    chirp_start_hz: float
    chirp_end_hz: float

    def __post_init__(self) -> None:
        values = (
            self.dt_ms,
            self.amplitude,
            self.flicker_hz,
            self.chirp_start_hz,
            self.chirp_end_hz,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise TemporalEvaluationError("Temporal probe values must be positive")
        if not 0 <= self.onset_step < self.offset_step <= self.time_steps:
            raise TemporalEvaluationError("Temporal probe window is invalid")
        if self.cone_count < 1:
            raise TemporalEvaluationError("Temporal probe needs at least one cone")


@dataclass(frozen=True, slots=True)
class TemporalMetricsRequest:
    response: torch.Tensor
    dt_ms: float
    onset_step: int
    offset_step: int


@dataclass(frozen=True, slots=True)
class TemporalResponseMetrics:
    response_latency_ms: float
    time_to_peak_ms: float
    crossover_ms: float | None
    recovery_ms: float | None
    transience_index: float


def build_temporal_probe(
    kind: TemporalProbeKind,
    spec: TemporalProbeSpec,
) -> torch.Tensor:
    stimulus = torch.zeros(spec.time_steps, spec.cone_count)
    active_steps = spec.offset_step - spec.onset_step
    elapsed_seconds = torch.arange(active_steps) * (spec.dt_ms / 1000.0)
    match kind:
        case TemporalProbeKind.IMPULSE:
            stimulus[spec.onset_step] = spec.amplitude
        case TemporalProbeKind.STEP:
            stimulus[spec.onset_step : spec.offset_step] = spec.amplitude
        case TemporalProbeKind.FLICKER:
            waveform = spec.amplitude * torch.sin(
                2.0 * torch.pi * spec.flicker_hz * elapsed_seconds
            )
            stimulus[spec.onset_step : spec.offset_step] = waveform[:, None]
        case TemporalProbeKind.CHIRP:
            duration = max(float(elapsed_seconds[-1]), spec.dt_ms / 1000.0)
            sweep = (spec.chirp_end_hz - spec.chirp_start_hz) / duration
            phase = 2.0 * torch.pi * (
                spec.chirp_start_hz * elapsed_seconds
                + 0.5 * sweep * elapsed_seconds.square()
            )
            stimulus[spec.onset_step : spec.offset_step] = (
                spec.amplitude * torch.sin(phase)
            )[:, None]
        case unreachable:
            assert_never(unreachable)
    return stimulus


def temporal_response_metrics(
    request: TemporalMetricsRequest,
) -> TemporalResponseMetrics:
    response = request.response.detach().to(dtype=torch.float32)
    if response.ndim != 1 or not torch.isfinite(response).all():
        raise TemporalEvaluationError("Temporal response must be a finite vector")
    if not math.isfinite(request.dt_ms) or request.dt_ms <= 0:
        raise TemporalEvaluationError("dt_ms must be positive and finite")
    if not 0 <= request.onset_step < request.offset_step <= response.shape[0]:
        raise TemporalEvaluationError("Temporal metric window is invalid")
    baseline = response[: request.onset_step].mean() if request.onset_step else 0.0
    centered = response - baseline
    post = centered[request.onset_step :]
    peak_offset = int(post.abs().argmax())
    peak = float(post[peak_offset])
    peak_magnitude = abs(peak)
    if peak_magnitude == 0.0:
        raise TemporalEvaluationError("Temporal response has no evoked component")
    threshold = 0.05 * peak_magnitude
    latency_offset = int(torch.nonzero(post.abs() >= threshold)[0])
    crossover = _crossover_ms(post, peak_offset, request.dt_ms)
    recovery = _recovery_ms(
        centered,
        request.offset_step,
        0.10 * peak_magnitude,
        request.dt_ms,
    )
    active = centered[request.onset_step : request.offset_step]
    late_count = max(1, math.ceil(active.numel() * 0.25))
    late_response = float(active[-late_count:].mean())
    peak_sign = 1.0 if peak > 0 else -1.0
    transience = 1.0 - peak_sign * late_response / peak_magnitude
    return TemporalResponseMetrics(
        response_latency_ms=latency_offset * request.dt_ms,
        time_to_peak_ms=peak_offset * request.dt_ms,
        crossover_ms=crossover,
        recovery_ms=recovery,
        transience_index=min(1.0, max(0.0, transience)),
    )


def _crossover_ms(
    response: torch.Tensor,
    peak_index: int,
    dt_ms: float,
) -> float | None:
    peak_sign = torch.sign(response[peak_index])
    for index in range(peak_index + 1, response.shape[0]):
        current_sign = torch.sign(response[index])
        if current_sign != 0 and current_sign != peak_sign:
            return index * dt_ms
    return None


def _recovery_ms(
    response: torch.Tensor,
    offset_step: int,
    tolerance: float,
    dt_ms: float,
) -> float | None:
    for index in range(offset_step, response.shape[0]):
        if torch.all(response[index:].abs() <= tolerance):
            return (index - offset_step) * dt_ms
    return None
