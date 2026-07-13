from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch

from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.retina_snn import RetinaSNNCore


class RGCPopulationName(StrEnum):
    MIDGET = "midget"
    PARASOL = "parasol"
    RESIDUAL = "residual"


class RFProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GradientRFRequest:
    x_cone: torch.Tensor
    population: RGCPopulationName
    polarity: int
    unit_index: int


@dataclass(frozen=True, slots=True)
class GradientRFResult:
    gradient: torch.Tensor
    response: torch.Tensor


@dataclass(frozen=True, slots=True)
class WhiteNoiseSTARequest:
    cone_count: int
    time_steps: int
    sample_count: int
    population: RGCPopulationName
    polarity: int
    unit_index: int
    seed: int = 7
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True, slots=True)
class WhiteNoiseSTAResult:
    sta: torch.Tensor
    response_mean: torch.Tensor
    response_std: torch.Tensor


@dataclass(frozen=True, slots=True)
class TemporalRFComparison:
    reference_ttp_ms: float
    condition_ttp_ms: float
    ttp_shift_ms: float
    peak_gain_ratio: float
    reference_biphasic_index: float
    condition_biphasic_index: float
    waveform_cosine_similarity: float


def gradient_rf(
    core: RetinaSNNCore,
    request: GradientRFRequest,
) -> GradientRFResult:
    x_cone = request.x_cone.detach().clone().requires_grad_(True)
    core.zero_grad(set_to_none=True)
    output, _ = core.forward_sequence(x_cone)
    response = _select_final_response(output, request)
    response.mean().backward()
    if x_cone.grad is None:
        raise RFProbeError("gradient RF did not produce input gradients")
    return GradientRFResult(
        gradient=x_cone.grad.detach().mean(dim=0),
        response=response.detach(),
    )


def white_noise_sta(
    core: RetinaSNNCore,
    request: WhiteNoiseSTARequest,
) -> WhiteNoiseSTAResult:
    generator = torch.Generator().manual_seed(request.seed)
    stimuli = torch.randn(
        request.sample_count,
        request.time_steps,
        request.cone_count,
        generator=generator,
        dtype=request.dtype,
    ).to(request.device)
    with torch.no_grad():
        output, _ = core.forward_sequence(stimuli)
        response = _select_final_response(
            output,
            GradientRFRequest(
                x_cone=stimuli,
                population=request.population,
                polarity=request.polarity,
                unit_index=request.unit_index,
            ),
        )
        centered = response - response.mean()
        denom = centered.abs().sum().clamp_min(torch.finfo(stimuli.dtype).eps)
        sta = (centered.view(-1, 1, 1) * stimuli).sum(dim=0) / denom
    return WhiteNoiseSTAResult(
        sta=sta,
        response_mean=response.mean(),
        response_std=response.std(unbiased=False),
    )


def compare_temporal_rfs(
    reference_rf: torch.Tensor,
    condition_rf: torch.Tensor,
    *,
    dt_ms: float,
) -> TemporalRFComparison:
    if reference_rf.shape != condition_rf.shape or reference_rf.ndim < 1:
        raise RFProbeError("temporal RFs must have the same non-empty shape [time,...]")
    if reference_rf.shape[0] < 1:
        raise RFProbeError("temporal RFs must contain at least one time step")
    if not torch.isfinite(reference_rf).all() or not torch.isfinite(condition_rf).all():
        raise RFProbeError("temporal RFs must be finite")
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise RFProbeError("dt_ms must be positive and finite")

    reference = reference_rf.detach().to(dtype=torch.float32).reshape(reference_rf.shape[0], -1)
    condition = condition_rf.detach().to(dtype=torch.float32).reshape(condition_rf.shape[0], -1)
    spatial_index = int(reference.abs().amax(dim=0).argmax())
    reference_trace = reference[:, spatial_index]
    condition_trace = condition[:, spatial_index]
    reference_peak = float(reference_trace.abs().max())
    if reference_peak == 0.0:
        raise RFProbeError("reference temporal RF must contain a non-zero response")
    condition_peak = float(condition_trace.abs().max())
    last_index = reference_trace.shape[0] - 1
    reference_ttp = (last_index - int(reference_trace.abs().argmax())) * dt_ms
    condition_ttp = (last_index - int(condition_trace.abs().argmax())) * dt_ms
    similarity = torch.nn.functional.cosine_similarity(
        reference_trace,
        condition_trace,
        dim=0,
    )
    return TemporalRFComparison(
        reference_ttp_ms=reference_ttp,
        condition_ttp_ms=condition_ttp,
        ttp_shift_ms=condition_ttp - reference_ttp,
        peak_gain_ratio=condition_peak / reference_peak,
        reference_biphasic_index=_biphasic_index(reference_trace),
        condition_biphasic_index=_biphasic_index(condition_trace),
        waveform_cosine_similarity=float(similarity),
    )


def _select_final_response(
    output: RGCOutput,
    request: GradientRFRequest,
) -> torch.Tensor:
    rates = _population_rates(output.rates, request.population)
    if rates.ndim != 4:
        raise RFProbeError("RF probe expects sequence rates [batch,time,2,N]")
    if request.polarity not in {0, 1}:
        raise RFProbeError("polarity must be 0 or 1")
    if request.unit_index < 0 or request.unit_index >= rates.shape[-1]:
        raise RFProbeError("unit_index is outside the selected population")
    return rates[:, -1, request.polarity, request.unit_index]


def _population_rates(
    populations: RGCPopulationTensors,
    name: RGCPopulationName,
) -> torch.Tensor:
    match name:
        case RGCPopulationName.MIDGET:
            return populations.midget
        case RGCPopulationName.PARASOL:
            return populations.parasol
        case RGCPopulationName.RESIDUAL:
            return populations.residual
        case unreachable:
            assert_never(unreachable)


def _biphasic_index(trace: torch.Tensor) -> float:
    positive_peak = max(float(trace.max()), 0.0)
    negative_peak = max(float(-trace.min()), 0.0)
    main_peak = max(positive_peak, negative_peak)
    return 0.0 if main_peak == 0.0 else min(positive_peak, negative_peak) / main_peak
