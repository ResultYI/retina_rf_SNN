from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from models.retina_snn import RetinaModel, RetinaState, detach_state


@dataclass(frozen=True, slots=True)
class MatchedContextPair:
    low_context: torch.Tensor
    high_context: torch.Tensor
    final_probe: torch.Tensor
    source_id: str


@dataclass(frozen=True, slots=True)
class PairStateCache:
    low: RetinaState
    high: RetinaState
    reset_a: RetinaState
    reset_b: RetinaState
    delayed: tuple[tuple[int, RetinaState, RetinaState], ...]


def build_state_cache(
    model: RetinaModel,
    pair: MatchedContextPair,
    probe: torch.Tensor,
    spatial_weights: torch.Tensor,
    delays_ms: Sequence[int],
    dt_ms: float,
) -> PairStateCache:
    device = probe.device
    low = _context_state(model, pair.low_context.unsqueeze(0).to(device), spatial_weights)
    high = _context_state(
        model,
        pair.high_context.unsqueeze(0).to(device),
        spatial_weights,
    )
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
                    neutral,
                    delayed_low,
                    spatial_weights=spatial_weights,
                )
                _, delayed_high = model.forward_sequence(
                    neutral,
                    delayed_high,
                    spatial_weights=spatial_weights,
                )
            delayed_low = detach_state(delayed_low)
            delayed_high = detach_state(delayed_high)
        delayed.append((int(delay_ms), delayed_low, delayed_high))
    return PairStateCache(low, high, reset_a, reset_b, tuple(delayed))


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


__all__ = ["MatchedContextPair", "PairStateCache", "build_state_cache"]
