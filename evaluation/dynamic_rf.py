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
    gain_normalized_cosine_distance: float
    temporal_peak_ms: float
    integration_width_ms: float
    spatial_second_moment: float
    state_reset_suppression: float
    recovery_curve: tuple[tuple[int, float], ...]
    finite_difference: FiniteDifferenceResult


def build_matched_context_pairs(
    clips: Sequence[PreparedClip],
    data_config: DataConfig,
    evaluation_config: EvaluationConfig,
) -> tuple[MatchedContextPair, ...]:
    if evaluation_config.dynamic_rf_lag_steps >= data_config.sequence_steps:
        raise DynamicRFError("dynamic RF lag must be shorter than the sequence")
    pairs: list[MatchedContextPair] = []
    for clip in clips[: evaluation_config.dynamic_rf_context_pairs]:
        clean = clip.clean
        split = clean.shape[0] - evaluation_config.dynamic_rf_lag_steps
        context = clean[:split]
        probe = clean[split:].clone()
        pairs.append(
            MatchedContextPair(
                low_context=context * data_config.context_gain_min,
                high_context=context * data_config.context_gain_max,
                final_probe=probe,
                source_id=clip.source_id,
            )
        )
    return tuple(pairs)


def evaluate_dynamic_rf(
    model: RetinaModel,
    pairs: Sequence[MatchedContextPair],
    config: EvaluationConfig,
    *,
    dt_ms: float,
    readout: Literal["spike_probability", "generator_potential"] = "spike_probability",
    finite_difference_epsilon: float = 1e-3,
) -> tuple[DynamicRFUnitResult, ...]:
    if not pairs:
        raise DynamicRFError("At least one matched context pair is required")
    device = next(model.parameters()).device
    spatial_weights = model.rgc.compute_spatial_weights().detach()
    results: list[DynamicRFUnitResult] = []
    for pair in pairs:
        low_context = pair.low_context.unsqueeze(0).to(device)
        high_context = pair.high_context.unsqueeze(0).to(device)
        probe = pair.final_probe.unsqueeze(0).to(device)
        low_state = _context_state(model, low_context, spatial_weights)
        high_state = _context_state(model, high_context, spatial_weights)
        with torch.no_grad():
            high_output, _ = model.forward_sequence(
                probe,
                high_state,
                spatial_weights=spatial_weights,
                probe_continuous_output=True,
            )
        scores = high_output.spike_probability.mean(dim=1)[0]
        for polarity in range(2):
            selected = torch.topk(
                scores[polarity],
                k=min(config.dynamic_rf_units_per_polarity, scores.shape[-1]),
            ).indices
            for unit_tensor in selected:
                unit = int(unit_tensor)
                low_kernel = _continuous_kernel(
                    model, probe, low_state, spatial_weights, polarity, unit, readout
                )
                high_kernel = _continuous_kernel(
                    model, probe, high_state, spatial_weights, polarity, unit, readout
                )
                normal_distance = float((low_kernel - high_kernel).norm())
                reset_kernel_a = _continuous_kernel(
                    model,
                    probe,
                    model.initial_state(1, device, probe.dtype),
                    spatial_weights,
                    polarity,
                    unit,
                    readout,
                )
                reset_kernel_b = _continuous_kernel(
                    model,
                    probe,
                    model.initial_state(1, device, probe.dtype),
                    spatial_weights,
                    polarity,
                    unit,
                    readout,
                )
                reset_distance = float((reset_kernel_a - reset_kernel_b).norm())
                suppression = 1.0 - reset_distance / (normal_distance + 1e-12)
                peak_ms, width_ms = _temporal_metrics(high_kernel, dt_ms)
                results.append(
                    DynamicRFUnitResult(
                        source_id=pair.source_id,
                        polarity=polarity,
                        unit=unit,
                        low_kernel_norm=float(low_kernel.norm()),
                        high_kernel_norm=float(high_kernel.norm()),
                        gain_normalized_cosine_distance=float(
                            1.0
                            - F.cosine_similarity(
                                low_kernel.flatten(), high_kernel.flatten(), dim=0
                            )
                        ),
                        temporal_peak_ms=peak_ms,
                        integration_width_ms=width_ms,
                        spatial_second_moment=_spatial_second_moment(
                            high_kernel, model.rgc.cone_positions_degs
                        ),
                        state_reset_suppression=suppression,
                        recovery_curve=_recovery_curve(
                            model,
                            probe,
                            low_state,
                            high_state,
                            spatial_weights,
                            polarity,
                            unit,
                            readout,
                            config.recovery_delays_ms,
                            dt_ms,
                        ),
                        finite_difference=_finite_difference_check(
                            model,
                            probe,
                            high_state,
                            spatial_weights,
                            polarity,
                            unit,
                            readout,
                            high_kernel,
                            finite_difference_epsilon,
                        ),
                    )
                )
    return tuple(results)


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
    gradient = torch.autograd.grad(continuous, differentiable_probe)[0]
    return gradient[0].detach()


def _finite_difference_check(
    model: RetinaModel,
    probe: torch.Tensor,
    state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    kernel: torch.Tensor,
    epsilon: float,
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
        events.append(output.hard_spikes.detach())
    autodiff = float((kernel * direction[0]).sum())
    finite_difference = float((outputs[0] - outputs[1]) / (2.0 * epsilon))
    if not torch.equal(events[0], events[1]):
        return FiniteDifferenceResult(
            autodiff,
            finite_difference,
            None,
            "threshold_crossing_not_local",
        )
    relative_error = abs(autodiff - finite_difference) / max(
        abs(autodiff), abs(finite_difference), 1e-12
    )
    return FiniteDifferenceResult(
        autodiff,
        finite_difference,
        relative_error,
        "local_continuous_check",
    )


def _temporal_metrics(kernel: torch.Tensor, dt_ms: float) -> tuple[float, float]:
    temporal = kernel.square().sum(dim=1).sqrt()
    peak_index = int(temporal.argmax())
    peak = temporal[peak_index].clamp_min(1e-12)
    width = int((temporal >= 0.5 * peak).sum())
    lag_to_peak = kernel.shape[0] - 1 - peak_index
    return lag_to_peak * dt_ms, width * dt_ms


def _spatial_second_moment(
    kernel: torch.Tensor,
    cone_positions: torch.Tensor,
) -> float:
    spatial = kernel.abs().sum(dim=0)
    normalized = spatial / spatial.sum().clamp_min(1e-12)
    center = (normalized[:, None] * cone_positions).sum(dim=0)
    return float(
        (normalized * (cone_positions - center).square().sum(dim=1)).sum()
    )


def _recovery_curve(
    model: RetinaModel,
    probe: torch.Tensor,
    low_state: RetinaState,
    high_state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    delays_ms: Sequence[int],
    dt_ms: float,
) -> tuple[tuple[int, float], ...]:
    curve: list[tuple[int, float]] = []
    for delay_ms in delays_ms:
        delay_steps = max(0, round(delay_ms / dt_ms))
        delayed_low = low_state
        delayed_high = high_state
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
        low_kernel = _continuous_kernel(
            model, probe, delayed_low, spatial_weights, polarity, unit, readout
        )
        high_kernel = _continuous_kernel(
            model, probe, delayed_high, spatial_weights, polarity, unit, readout
        )
        curve.append((int(delay_ms), float((low_kernel - high_kernel).norm())))
    return tuple(curve)


__all__ = [
    "DynamicRFError",
    "DynamicRFUnitResult",
    "FiniteDifferenceResult",
    "MatchedContextPair",
    "build_matched_context_pairs",
    "evaluate_dynamic_rf",
]
