from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict, assert_never

import numpy as np
import torch

from evaluation.checkpoint_metrics import HeldOutEvaluation
from evaluation.dynamics import (
    TemporalEvaluationError,
    TemporalMetricsRequest,
    TemporalProbeKind,
    TemporalProbeSpec,
    build_temporal_probe,
    temporal_response_metrics,
)
from evaluation.rf_agreement import RFMapAgreementError, compare_rf_maps
from evaluation.rf_probe import (
    GradientRFRequest,
    LocalPoissonGLMRequest,
    RGCPopulationName,
    WhiteNoiseSTARequest,
    fit_local_poisson_glm,
    gradient_rf,
    white_noise_sta,
)
from models.cells.rgc import RGCPopulationTensors
from models.retina_snn import RetinaSNNCore
from training.stage1 import Stage1Components


class RFProbeMetrics(TypedDict):
    population: str
    polarity: int
    status: str
    jacobian_response_mean: float
    sta_response_mean: float
    sta_response_std: float
    glm_nll: float
    jacobian_sta_cosine: float | None
    jacobian_glm_cosine: float | None
    sta_glm_cosine: float | None


class TemporalProbeMetrics(TypedDict):
    kind: str
    population: str
    polarity: int
    status: str
    response_latency_ms: float | None
    time_to_peak_ms: float | None
    crossover_ms: float | None
    recovery_ms: float | None
    transience_index: float | None


@dataclass(frozen=True, slots=True)
class RFProbeRequest:
    components: Stage1Components
    held_out: HeldOutEvaluation
    sample_count: int
    glm_max_steps: int


@dataclass(frozen=True, slots=True)
class RFProbeBundle:
    metrics: tuple[RFProbeMetrics, ...]
    arrays: dict[str, np.ndarray]


def run_rf_probes(request: RFProbeRequest) -> RFProbeBundle:
    components = request.components
    positions = torch.as_tensor(
        components.mosaic.bipolar_positions_degs,
        device=request.held_out.probe_stimuli.device,
        dtype=torch.float32,
    )
    arrays: dict[str, np.ndarray] = {}
    summaries = []
    for population in RGCPopulationName:
        source_indices = _source_indices(components, population)
        for polarity in (0, 1):
            name = f"{population.value}_{polarity}"
            gradient = gradient_rf(
                components.core,
                GradientRFRequest(
                    request.held_out.probe_stimuli,
                    population,
                    polarity,
                    0,
                ),
            )
            sta = white_noise_sta(
                components.core,
                WhiteNoiseSTARequest(
                    cone_count=positions.shape[0],
                    time_steps=request.held_out.probe_stimuli.shape[1],
                    sample_count=request.sample_count,
                    population=population,
                    polarity=polarity,
                    unit_index=0,
                    device=request.held_out.probe_stimuli.device,
                ),
            )
            counts = _population_values(
                request.held_out.probe_output.spikes,
                population,
            )[:, :, polarity, 0].sum(dim=1)
            glm = fit_local_poisson_glm(
                LocalPoissonGLMRequest(
                    stimulus=request.held_out.probe_stimuli,
                    spike_counts=counts,
                    source_indices=source_indices,
                    max_steps=request.glm_max_steps,
                )
            )
            agreements = (
                _cosine(gradient.gradient, sta.sta, positions),
                _cosine(gradient.gradient, glm.rf, positions),
                _cosine(sta.sta, glm.rf, positions),
            )
            status = (
                "ok" if all(value is not None for value in agreements) else "partial"
            )
            summaries.append(
                RFProbeMetrics(
                    population=population.value,
                    polarity=polarity,
                    status=status,
                    jacobian_response_mean=gradient.response.mean().item(),
                    sta_response_mean=sta.response_mean.item(),
                    sta_response_std=sta.response_std.item(),
                    glm_nll=glm.nll.item(),
                    jacobian_sta_cosine=agreements[0],
                    jacobian_glm_cosine=agreements[1],
                    sta_glm_cosine=agreements[2],
                )
            )
            arrays[f"{name}_jacobian"] = _numpy(gradient.gradient)
            arrays[f"{name}_sta"] = _numpy(sta.sta)
            arrays[f"{name}_glm"] = _numpy(glm.rf)
    return RFProbeBundle(tuple(summaries), arrays)


def run_temporal_probes(
    core: RetinaSNNCore,
    cone_count: int,
    dt_ms: float,
) -> tuple[TemporalProbeMetrics, ...]:
    time_steps = max(20, math.ceil(1000.0 / dt_ms))
    onset = max(1, time_steps // 5)
    offset = max(onset + 1, 7 * time_steps // 10)
    spec = TemporalProbeSpec(
        cone_count=cone_count,
        time_steps=time_steps,
        dt_ms=dt_ms,
        onset_step=onset,
        offset_step=offset,
        amplitude=1.0,
        flicker_hz=4.0,
        chirp_start_hz=0.5,
        chirp_end_hz=8.0,
    )
    device = next(core.parameters()).device
    results = []
    core.eval()
    with torch.no_grad():
        for kind in TemporalProbeKind:
            stimulus = build_temporal_probe(kind, spec)[None].to(device)
            output, _ = core.forward_sequence(stimulus)
            for population in RGCPopulationName:
                rates = _population_values(output.rates, population)
                for polarity in (0, 1):
                    response = rates[0, :, polarity].mean(dim=-1)
                    results.append(
                        _temporal_metrics(kind, population, polarity, response, spec)
                    )
    return tuple(results)


def _temporal_metrics(
    kind: TemporalProbeKind,
    population: RGCPopulationName,
    polarity: int,
    response: torch.Tensor,
    spec: TemporalProbeSpec,
) -> TemporalProbeMetrics:
    try:
        metrics = temporal_response_metrics(
            TemporalMetricsRequest(
                response=response,
                dt_ms=spec.dt_ms,
                onset_step=spec.onset_step,
                offset_step=spec.offset_step,
            )
        )
    except TemporalEvaluationError:
        return TemporalProbeMetrics(
            kind=kind.value,
            population=population.value,
            polarity=polarity,
            status="no_evoked_response",
            response_latency_ms=None,
            time_to_peak_ms=None,
            crossover_ms=None,
            recovery_ms=None,
            transience_index=None,
        )
    return TemporalProbeMetrics(
        kind=kind.value,
        population=population.value,
        polarity=polarity,
        status="ok",
        response_latency_ms=metrics.response_latency_ms,
        time_to_peak_ms=metrics.time_to_peak_ms,
        crossover_ms=metrics.crossover_ms,
        recovery_ms=metrics.recovery_ms,
        transience_index=metrics.transience_index,
    )


def _source_indices(
    components: Stage1Components,
    population: RGCPopulationName,
) -> torch.Tensor:
    match population:
        case RGCPopulationName.MIDGET:
            pool = components.core.rgc.midget_pool
        case RGCPopulationName.PARASOL:
            pool = components.core.rgc.parasol_pool
        case RGCPopulationName.RESIDUAL:
            pool = components.core.rgc.residual_pool
        case unreachable:
            assert_never(unreachable)
    indices = pool.coalesce().indices()
    return indices[1, indices[0] == 0]


def _population_values(
    values: RGCPopulationTensors,
    population: RGCPopulationName,
) -> torch.Tensor:
    match population:
        case RGCPopulationName.MIDGET:
            return values.midget
        case RGCPopulationName.PARASOL:
            return values.parasol
        case RGCPopulationName.RESIDUAL:
            return values.residual
        case unreachable:
            assert_never(unreachable)


def _cosine(
    first: torch.Tensor,
    second: torch.Tensor,
    positions: torch.Tensor,
) -> float | None:
    try:
        return compare_rf_maps(first, second, positions).cosine_similarity
    except RFMapAgreementError:
        return None


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()
