from __future__ import annotations

import torch

from models.mechanistic_retina.temporal_parameters import (
    bounded_tau_ms,
    ordered_bounded_tau_ms,
    raw_ordered_tau_from_ms,
    raw_tau_from_ms,
)


def raw_delay_from_ms(
    initial_ms: torch.Tensor, bounds_ms: torch.Tensor
) -> torch.Tensor:
    return raw_tau_from_ms(initial_ms, bounds_ms)


def bounded_delay_ms(raw_delay: torch.Tensor, bounds_ms: torch.Tensor) -> torch.Tensor:
    return bounded_tau_ms(raw_delay, bounds_ms)


def raw_ordered_delay_from_ms(
    initial_ms: torch.Tensor, bounds_ms: torch.Tensor
) -> torch.Tensor:
    return raw_ordered_tau_from_ms(initial_ms, bounds_ms)


def ordered_bounded_delay_ms(
    raw_delay: torch.Tensor, bounds_ms: torch.Tensor
) -> torch.Tensor:
    return ordered_bounded_tau_ms(raw_delay, bounds_ms)


__all__ = [
    "bounded_delay_ms",
    "ordered_bounded_delay_ms",
    "raw_delay_from_ms",
    "raw_ordered_delay_from_ms",
]
