from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch.nn import functional as F

from evaluation.dynamic_rf_metrics import (
    FiniteDifferenceResult,
    continuous_kernel,
    finite_difference_check,
    spatial_metrics,
    temporal_metrics,
)
from evaluation.dynamic_rf_state import (
    MatchedContextPair,
    PairStateCache,
    build_state_cache,
)
from models.retina_snn import RetinaModel
from training.config import DataConfig, EvaluationConfig
from training.data import PreparedClip


class DynamicRFError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DynamicRFSelection:
    polarity: int
    unit_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DynamicRFUnitResult:
    source_id: str
    polarity: int
    unit: int
    low_kernel_norm: float
    high_kernel_norm: float
    kernel_norm_ratio: float
    gain_normalized_cosine_distance: float
    low_temporal_peak_ms: float
    high_temporal_peak_ms: float
    temporal_peak_shift_ms: float
    low_integration_width_ms: float
    high_integration_width_ms: float
    integration_width_shift_ms: float
    low_spatial_center_distance_degs: float
    high_spatial_center_distance_degs: float
    spatial_center_shift_degs: float
    low_spatial_second_moment: float
    high_spatial_second_moment: float
    spatial_second_moment_shift: float
    identical_reset_kernel_error: float
    recovery_curve: tuple[tuple[int, float], ...]
    finite_difference: FiniteDifferenceResult


def build_matched_context_pairs(
    clips: Sequence[PreparedClip],
    data_config: DataConfig,
    evaluation_config: EvaluationConfig,
) -> tuple[MatchedContextPair, ...]:
    if evaluation_config.dynamic_rf_lag_steps >= data_config.sequence_steps:
        raise DynamicRFError("dynamic RF lag must be shorter than the sequence")
    return tuple(
        MatchedContextPair(
            low_context=clip.clean[: -evaluation_config.dynamic_rf_lag_steps]
            * data_config.context_gain_min,
            high_context=clip.clean[: -evaluation_config.dynamic_rf_lag_steps]
            * data_config.context_gain_max,
            final_probe=clip.clean[-evaluation_config.dynamic_rf_lag_steps :].clone(),
            source_id=clip.source_id,
        )
        for clip in clips[: evaluation_config.dynamic_rf_max_sources]
    )


@torch.no_grad()
def select_dynamic_rf_units(
    trained_model: RetinaModel,
    pairs: Sequence[MatchedContextPair],
    config: EvaluationConfig,
) -> tuple[DynamicRFSelection, ...]:
    if not pairs:
        raise DynamicRFError("At least one matched context pair is required")
    device = next(trained_model.parameters()).device
    spatial_weights = trained_model.rgc.compute_spatial_weights().detach()
    scores = torch.zeros(
        2,
        trained_model.rgc.unit_count,
        device=device,
    )
    was_training = trained_model.training
    trained_model.eval()
    try:
        for pair in pairs:
            high_state = _context_state(
                trained_model,
                pair.high_context.unsqueeze(0).to(device),
                spatial_weights,
            )
            output, _ = trained_model.forward_sequence(
                pair.final_probe.unsqueeze(0).to(device),
                high_state,
                spatial_weights=spatial_weights,
                probe_continuous_output=True,
            )
            scores += output.spike_probability.mean(dim=1)[0]
    finally:
        trained_model.train(was_training)
    count = min(config.dynamic_rf_units_per_polarity, scores.shape[-1])
    return tuple(
        DynamicRFSelection(
            polarity=polarity,
            unit_indices=tuple(int(unit) for unit in torch.topk(scores[polarity], count).indices),
        )
        for polarity in range(2)
    )


def evaluate_dynamic_rf(
    model: RetinaModel,
    pairs: Sequence[MatchedContextPair],
    config: EvaluationConfig,
    *,
    dt_ms: float,
    selection_plan: Sequence[DynamicRFSelection] | None = None,
    readout: Literal["spike_probability", "generator_potential"] = "spike_probability",
    finite_difference_epsilons: Sequence[float] = (1e-4, 3e-4, 1e-3),
) -> tuple[DynamicRFUnitResult, ...]:
    if not pairs:
        raise DynamicRFError("At least one matched context pair is required")
    selections = tuple(selection_plan or select_dynamic_rf_units(model, pairs, config))
    if {selection.polarity for selection in selections} != {0, 1}:
        raise DynamicRFError("selection_plan must contain ON and OFF selections")
    device = next(model.parameters()).device
    spatial_weights = model.rgc.compute_spatial_weights().detach()
    results: list[DynamicRFUnitResult] = []
    was_training = model.training
    model.eval()
    try:
        for pair in pairs:
            probe = pair.final_probe.unsqueeze(0).to(device)
            cache = build_state_cache(
                model,
                pair,
                probe,
                spatial_weights,
                config.recovery_delays_ms,
                dt_ms,
            )
            for selection in selections:
                for unit in selection.unit_indices:
                    results.append(
                        _evaluate_unit(
                            model,
                            pair.source_id,
                            probe,
                            cache,
                            spatial_weights,
                            selection.polarity,
                            unit,
                            readout,
                            finite_difference_epsilons,
                            dt_ms,
                        )
                    )
    finally:
        model.train(was_training)
    return tuple(results)


def _evaluate_unit(
    model: RetinaModel,
    source_id: str,
    probe: torch.Tensor,
    cache: PairStateCache,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    finite_difference_epsilons: Sequence[float],
    dt_ms: float,
) -> DynamicRFUnitResult:
    low_kernel = continuous_kernel(
        model, probe, cache.low, spatial_weights, polarity, unit, readout
    )
    high_kernel = continuous_kernel(
        model, probe, cache.high, spatial_weights, polarity, unit, readout
    )
    reset_a = continuous_kernel(
        model, probe, cache.reset_a, spatial_weights, polarity, unit, readout
    )
    reset_b = continuous_kernel(
        model, probe, cache.reset_b, spatial_weights, polarity, unit, readout
    )
    low_peak_ms, low_width_ms = temporal_metrics(low_kernel, dt_ms)
    high_peak_ms, high_width_ms = temporal_metrics(high_kernel, dt_ms)
    unit_center = model.rgc.unit_centers_degs[unit]
    low_center, low_center_distance, low_moment = spatial_metrics(
        low_kernel, model.rgc.cone_positions_degs, unit_center
    )
    high_center, high_center_distance, high_moment = spatial_metrics(
        high_kernel, model.rgc.cone_positions_degs, unit_center
    )
    low_norm = float(low_kernel.norm())
    high_norm = float(high_kernel.norm())
    recovery: list[tuple[int, float]] = []
    for delay_ms, delayed_low, delayed_high in cache.delayed:
        if delay_ms == 0:
            delayed_low_kernel = low_kernel
            delayed_high_kernel = high_kernel
        else:
            delayed_low_kernel = continuous_kernel(
                model, probe, delayed_low, spatial_weights, polarity, unit, readout
            )
            delayed_high_kernel = continuous_kernel(
                model, probe, delayed_high, spatial_weights, polarity, unit, readout
            )
        recovery.append(
            (delay_ms, float((delayed_low_kernel - delayed_high_kernel).norm()))
        )
    return DynamicRFUnitResult(
        source_id=source_id,
        polarity=polarity,
        unit=unit,
        low_kernel_norm=low_norm,
        high_kernel_norm=high_norm,
        kernel_norm_ratio=high_norm / max(low_norm, 1e-12),
        gain_normalized_cosine_distance=float(
            1.0 - F.cosine_similarity(low_kernel.flatten(), high_kernel.flatten(), dim=0)
        ),
        low_temporal_peak_ms=low_peak_ms,
        high_temporal_peak_ms=high_peak_ms,
        temporal_peak_shift_ms=high_peak_ms - low_peak_ms,
        low_integration_width_ms=low_width_ms,
        high_integration_width_ms=high_width_ms,
        integration_width_shift_ms=high_width_ms - low_width_ms,
        low_spatial_center_distance_degs=low_center_distance,
        high_spatial_center_distance_degs=high_center_distance,
        spatial_center_shift_degs=float((high_center - low_center).norm()),
        low_spatial_second_moment=low_moment,
        high_spatial_second_moment=high_moment,
        spatial_second_moment_shift=high_moment - low_moment,
        identical_reset_kernel_error=float((reset_a - reset_b).norm()),
        recovery_curve=tuple(recovery),
        finite_difference=finite_difference_check(
            model,
            probe,
            cache.high,
            spatial_weights,
            polarity,
            unit,
            readout,
            high_kernel,
            finite_difference_epsilons,
        ),
    )


__all__ = (
    "DynamicRFError DynamicRFSelection DynamicRFUnitResult FiniteDifferenceResult "
    "MatchedContextPair build_matched_context_pairs evaluate_dynamic_rf select_dynamic_rf_units"
).split()
