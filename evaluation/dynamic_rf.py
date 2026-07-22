from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch.nn import functional as F

from models.retina_snn import RetinaModel, RetinaState, detach_state
from training.config import DataConfig, EvaluationConfig
from training.data import PreparedClip


class DynamicRFError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MatchedContextPair:
    low_context: torch.Tensor
    high_context: torch.Tensor
    final_probe: torch.Tensor
    source_id: str


@dataclass(frozen=True, slots=True)
class DynamicRFSelection:
    polarity: int
    unit_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FiniteDifferenceResult:
    autodiff_directional: float
    finite_difference_directional: float
    relative_error: float | None
    status: str


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


@dataclass(frozen=True, slots=True)
class _PairStateCache:
    low: RetinaState
    high: RetinaState
    reset_a: RetinaState
    reset_b: RetinaState
    delayed: tuple[tuple[int, RetinaState, RetinaState], ...]


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
            cache = _build_state_cache(
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


def _build_state_cache(
    model: RetinaModel,
    pair: MatchedContextPair,
    probe: torch.Tensor,
    spatial_weights: torch.Tensor,
    delays_ms: Sequence[int],
    dt_ms: float,
) -> _PairStateCache:
    device = probe.device
    low = _context_state(model, pair.low_context.unsqueeze(0).to(device), spatial_weights)
    high = _context_state(model, pair.high_context.unsqueeze(0).to(device), spatial_weights)
    reset_a = model.initial_state(1, device, probe.dtype)
    reset_b = model.initial_state(1, device, probe.dtype)
    delayed: list[tuple[int, RetinaState, RetinaState]] = []
    for delay_ms in delays_ms:
        delay_steps = max(0, round(delay_ms / dt_ms))
        delayed_low, delayed_high = low, high
        if delay_steps:
            neutral = probe.new_zeros((1, delay_steps, probe.shape[-1]))
            with torch.no_grad():
                _, delayed_low = model.forward_sequence(
                    neutral, delayed_low, spatial_weights=spatial_weights
                )
                _, delayed_high = model.forward_sequence(
                    neutral, delayed_high, spatial_weights=spatial_weights
                )
            delayed_low = detach_state(delayed_low)
            delayed_high = detach_state(delayed_high)
        delayed.append((int(delay_ms), delayed_low, delayed_high))
    return _PairStateCache(low, high, reset_a, reset_b, tuple(delayed))


def _evaluate_unit(
    model: RetinaModel,
    source_id: str,
    probe: torch.Tensor,
    cache: _PairStateCache,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    finite_difference_epsilons: Sequence[float],
    dt_ms: float,
) -> DynamicRFUnitResult:
    low_kernel = _continuous_kernel(
        model, probe, cache.low, spatial_weights, polarity, unit, readout
    )
    high_kernel = _continuous_kernel(
        model, probe, cache.high, spatial_weights, polarity, unit, readout
    )
    reset_a = _continuous_kernel(
        model, probe, cache.reset_a, spatial_weights, polarity, unit, readout
    )
    reset_b = _continuous_kernel(
        model, probe, cache.reset_b, spatial_weights, polarity, unit, readout
    )
    low_peak_ms, low_width_ms = _temporal_metrics(low_kernel, dt_ms)
    high_peak_ms, high_width_ms = _temporal_metrics(high_kernel, dt_ms)
    unit_center = model.rgc.unit_centers_degs[unit]
    low_center, low_center_distance, low_moment = _spatial_metrics(
        low_kernel, model.rgc.cone_positions_degs, unit_center
    )
    high_center, high_center_distance, high_moment = _spatial_metrics(
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
            delayed_low_kernel = _continuous_kernel(
                model, probe, delayed_low, spatial_weights, polarity, unit, readout
            )
            delayed_high_kernel = _continuous_kernel(
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
        finite_difference=_finite_difference_check(
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


def _context_state(
    model: RetinaModel,
    context: torch.Tensor,
    spatial_weights: torch.Tensor,
) -> RetinaState:
    with torch.no_grad():
        _, state = model.forward_sequence(
            context,
            spatial_weights=spatial_weights,
            probe_continuous_output=True,
        )
    return detach_state(state)


def _continuous_kernel(
    model: RetinaModel,
    probe: torch.Tensor,
    state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
) -> torch.Tensor:
    differentiable_probe = probe.detach().clone().requires_grad_(True)
    output, _ = model.forward_sequence(
        differentiable_probe,
        state,
        spatial_weights=spatial_weights,
        probe_continuous_output=True,
    )
    continuous = getattr(output, readout)[0, -1, polarity, unit]
    return torch.autograd.grad(continuous, differentiable_probe)[0][0].detach()


def _finite_difference_check(
    model: RetinaModel,
    probe: torch.Tensor,
    state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    kernel: torch.Tensor,
    epsilons: Sequence[float],
) -> FiniteDifferenceResult:
    generator = torch.Generator(device=probe.device).manual_seed(
        polarity * model.rgc.unit_count + unit
    )
    direction = torch.randn(
        probe.shape,
        generator=generator,
        device=probe.device,
        dtype=probe.dtype,
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    autodiff = float((kernel * direction[0]).sum())
    last_finite_difference = 0.0
    with torch.no_grad():
        for epsilon in sorted(epsilons, reverse=True):
            outputs = []
            events = []
            for signed_epsilon in (epsilon, -epsilon):
                output, _ = model.forward_sequence(
                    probe + signed_epsilon * direction,
                    state,
                    spatial_weights=spatial_weights,
                    probe_continuous_output=True,
                )
                outputs.append(getattr(output, readout)[0, -1, polarity, unit])
                events.append(output.hard_spikes[0, :, polarity, unit])
            last_finite_difference = float((outputs[0] - outputs[1]) / (2.0 * epsilon))
            if torch.equal(events[0], events[1]):
                relative_error = abs(autodiff - last_finite_difference) / max(
                    abs(autodiff), abs(last_finite_difference), 1e-12
                )
                return FiniteDifferenceResult(
                    autodiff,
                    last_finite_difference,
                    relative_error,
                    "local_continuous_check",
                )
    return FiniteDifferenceResult(
        autodiff,
        last_finite_difference,
        None,
        "threshold_crossing_not_local",
    )


def _temporal_metrics(kernel: torch.Tensor, dt_ms: float) -> tuple[float, float]:
    temporal = kernel.square().sum(dim=1).sqrt()
    peak_index = int(temporal.argmax())
    peak = temporal[peak_index].clamp_min(1e-12)
    width = int((temporal >= 0.5 * peak).sum())
    return (kernel.shape[0] - 1 - peak_index) * dt_ms, width * dt_ms


def _spatial_metrics(
    kernel: torch.Tensor,
    cone_positions: torch.Tensor,
    unit_center: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    spatial = kernel.abs().sum(dim=0)
    normalized = spatial / spatial.sum().clamp_min(1e-12)
    center = (normalized[:, None] * cone_positions).sum(dim=0)
    second_moment = (
        normalized * (cone_positions - unit_center).square().sum(dim=1)
    ).sum()
    return center, float((center - unit_center).norm()), float(second_moment)


__all__ = [
    "DynamicRFError",
    "DynamicRFSelection",
    "DynamicRFUnitResult",
    "FiniteDifferenceResult",
    "MatchedContextPair",
    "build_matched_context_pairs",
    "evaluate_dynamic_rf",
    "select_dynamic_rf_units",
]
