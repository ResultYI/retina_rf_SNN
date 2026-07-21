from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, assert_never

import numpy as np
import torch

from evaluation.checkpoint_metrics import HeldOutEvaluation
from evaluation.rf_agreement import RFMapAgreementError, compare_rf_maps
from evaluation.rf_probe import (
    GradientRFRequest,
    LocalPoissonGLMRequest,
    RFProbeError,
    RGCPopulationName,
    WhiteNoiseSTARequest,
    compare_temporal_rfs,
    fit_local_poisson_glm,
    gradient_rf,
    white_noise_sta,
)
from evaluation.temporal_probes import TemporalProbeMetrics, run_temporal_probes
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from training.stage1 import Stage1Components


_PROBE_POPULATIONS = (
    RGCPopulationName.MIDGET,
    RGCPopulationName.PARASOL,
)


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
    low_high_context_waveform_cosine: float | None
    low_high_context_ttp_shift_ms: float | None
    low_high_context_peak_gain_ratio: float | None


@dataclass(frozen=True, slots=True)
class RFProbeRequest:
    components: Stage1Components
    held_out: HeldOutEvaluation
    probe_stimuli: torch.Tensor
    probe_output: RGCOutput
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
        device=request.probe_stimuli.device,
        dtype=torch.float32,
    )
    arrays: dict[str, np.ndarray] = {}
    summaries = []
    contrast_contexts = _contrast_contexts(request.probe_stimuli)
    for population in _PROBE_POPULATIONS:
        source_indices = _source_indices(components, population)
        for polarity in (0, 1):
            name = f"{population.value}_{polarity}"
            gradient = gradient_rf(
                components.core,
                GradientRFRequest(
                    request.probe_stimuli,
                    population,
                    polarity,
                    0,
                ),
            )
            conditioned = _conditioned_gradient_comparison(
                components,
                contrast_contexts,
                population,
                polarity,
            )
            sta = white_noise_sta(
                components.core,
                WhiteNoiseSTARequest(
                    cone_count=positions.shape[0],
                    time_steps=request.probe_stimuli.shape[1],
                    sample_count=request.sample_count,
                    population=population,
                    polarity=polarity,
                    unit_index=0,
                    device=request.probe_stimuli.device,
                ),
            )
            counts = _population_values(
                request.probe_output.spikes,
                population,
            )[:, :, polarity, 0].sum(dim=1)
            glm = fit_local_poisson_glm(
                LocalPoissonGLMRequest(
                    stimulus=request.probe_stimuli,
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
                    low_high_context_waveform_cosine=(
                        None if conditioned is None else conditioned[2]
                    ),
                    low_high_context_ttp_shift_ms=(
                        None if conditioned is None else conditioned[3]
                    ),
                    low_high_context_peak_gain_ratio=(
                        None if conditioned is None else conditioned[4]
                    ),
                )
            )
            arrays[f"{name}_jacobian"] = _numpy(gradient.gradient)
            arrays[f"{name}_sta"] = _numpy(sta.sta)
            arrays[f"{name}_glm"] = _numpy(glm.rf)
            if conditioned is not None:
                arrays[f"{name}_jacobian_low_context"] = _numpy(conditioned[0])
                arrays[f"{name}_jacobian_high_context"] = _numpy(conditioned[1])
    return RFProbeBundle(tuple(summaries), arrays)


def _source_indices(
    components: Stage1Components,
    population: RGCPopulationName,
) -> torch.Tensor:
    match population:
        case RGCPopulationName.MIDGET:
            pool = components.core.rgc.midget_pool
        case RGCPopulationName.PARASOL:
            pool = components.core.rgc.parasol_pool
        case unreachable:
            assert_never(unreachable)
    indices = pool.coalesce().indices()
    return indices[1, indices[0] == 0]


def _contrast_contexts(
    stimuli: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = 0.5 * stimuli
    low[:, -1] = stimuli[:, -1]
    return low, stimuli


def _conditioned_gradient_comparison(
    components: Stage1Components,
    groups: tuple[torch.Tensor, torch.Tensor] | None,
    population: RGCPopulationName,
    polarity: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float] | None:
    if groups is None:
        return None
    low = gradient_rf(
        components.core,
        GradientRFRequest(groups[0], population, polarity, 0),
    ).gradient
    high = gradient_rf(
        components.core,
        GradientRFRequest(groups[1], population, polarity, 0),
    ).gradient
    try:
        comparison = compare_temporal_rfs(
            low,
            high,
            dt_ms=components.profile.rgc.dt_ms,
        )
    except RFProbeError:
        return None
    return (
        low,
        high,
        comparison.waveform_cosine_similarity,
        comparison.ttp_shift_ms,
        comparison.peak_gain_ratio,
    )


def _population_values(
    values: RGCPopulationTensors,
    population: RGCPopulationName,
) -> torch.Tensor:
    match population:
        case RGCPopulationName.MIDGET:
            return values.midget
        case RGCPopulationName.PARASOL:
            return values.parasol
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
