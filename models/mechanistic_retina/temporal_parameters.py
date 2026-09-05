from __future__ import annotations

import torch


class TemporalParameterError(ValueError):
    pass


def raw_tau_from_ms(initial_ms: torch.Tensor, bounds_ms: torch.Tensor) -> torch.Tensor:
    _validate_shapes(initial_ms, bounds_ms)
    lower = bounds_ms[..., 0]
    upper = bounds_ms[..., 1]
    if not bool(torch.isfinite(initial_ms).all() and torch.isfinite(bounds_ms).all()):
        raise TemporalParameterError("temporal parameters and bounds must be finite")
    if not bool(torch.all(lower < initial_ms) and torch.all(initial_ms < upper)):
        raise TemporalParameterError("initial temporal parameters must lie inside bounds")
    fraction = (initial_ms - lower) / (upper - lower)
    return torch.logit(fraction)


def bounded_tau_ms(raw_tau: torch.Tensor, bounds_ms: torch.Tensor) -> torch.Tensor:
    lower = bounds_ms[..., 0]
    return lower + (bounds_ms[..., 1] - lower) * torch.sigmoid(raw_tau)


def raw_ordered_tau_from_ms(
    initial_ms: torch.Tensor, bounds_ms: torch.Tensor
) -> torch.Tensor:
    _validate_ordered_pair(initial_ms, bounds_ms)
    sustained_initial, transient_initial = initial_ms.unbind()
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    epsilon = torch.finfo(bounds_ms.dtype).eps * sustained_bounds[..., 1]
    transient_upper = torch.minimum(
        transient_bounds[..., 1], sustained_bounds[..., 1] - epsilon
    )
    transient_fraction = (transient_initial - transient_bounds[..., 0]) / (
        transient_upper - transient_bounds[..., 0]
    )
    sustained_lower = torch.maximum(
        sustained_bounds[..., 0], transient_initial + epsilon
    )
    sustained_fraction = (sustained_initial - sustained_lower) / (
        sustained_bounds[..., 1] - sustained_lower
    )
    return torch.stack(
        (torch.logit(sustained_fraction), torch.logit(transient_fraction))
    )


def ordered_bounded_tau_ms(
    raw_tau: torch.Tensor, bounds_ms: torch.Tensor
) -> torch.Tensor:
    sustained_raw, transient_raw = raw_tau.unbind()
    sustained_bounds, transient_bounds = bounds_ms.unbind()
    epsilon = torch.finfo(bounds_ms.dtype).eps * sustained_bounds[..., 1]
    transient_upper = torch.minimum(
        transient_bounds[..., 1], sustained_bounds[..., 1] - epsilon
    )
    transient = transient_bounds[..., 0] + (
        transient_upper - transient_bounds[..., 0]
    ) * torch.sigmoid(transient_raw)
    sustained_lower = torch.maximum(
        sustained_bounds[..., 0], transient + epsilon
    )
    sustained = sustained_lower + (
        sustained_bounds[..., 1] - sustained_lower
    ) * torch.sigmoid(sustained_raw)
    return torch.stack((sustained, transient))


def _validate_shapes(values: torch.Tensor, bounds_ms: torch.Tensor) -> None:
    if bounds_ms.shape != values.shape + (2,):
        raise TemporalParameterError(
            "temporal bounds must have shape tau_shape + (lower, upper)"
        )
    if not bool(torch.all(bounds_ms[..., 0] < bounds_ms[..., 1])):
        raise TemporalParameterError("temporal lower bounds must be below upper bounds")


def _validate_ordered_pair(
    initial_ms: torch.Tensor, bounds_ms: torch.Tensor
) -> None:
    _validate_shapes(initial_ms, bounds_ms)
    if initial_ms.shape[0] != 2:
        raise TemporalParameterError(
            "ordered temporal parameters must start with sustained/local and transient"
        )
    if not bool(torch.isfinite(initial_ms).all() and torch.isfinite(bounds_ms).all()):
        raise TemporalParameterError("ordered temporal parameters must be finite")
    sustained, transient = initial_ms.unbind()
    if not bool(torch.all(sustained > transient)):
        raise TemporalParameterError(
            "sustained/local tau must be greater than transient tau"
        )
    lower = bounds_ms[..., 0]
    upper = bounds_ms[..., 1]
    if not bool(torch.all(lower < initial_ms) and torch.all(initial_ms < upper)):
        raise TemporalParameterError(
            "initial ordered temporal parameters must lie inside bounds"
        )
    epsilon = torch.finfo(bounds_ms.dtype).eps * bounds_ms[0, ..., 1]
    transient_upper = torch.minimum(
        bounds_ms[1, ..., 1], bounds_ms[0, ..., 1] - epsilon
    )
    if not bool(torch.all(bounds_ms[1, ..., 0] < transient_upper)):
        raise TemporalParameterError("ordered temporal bounds cannot preserve ordering")


__all__ = [
    "TemporalParameterError",
    "bounded_tau_ms",
    "ordered_bounded_tau_ms",
    "raw_ordered_tau_from_ms",
    "raw_tau_from_ms",
]
