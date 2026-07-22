from __future__ import annotations

from dataclasses import dataclass

import torch

from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel


@dataclass(frozen=True, slots=True)
class ParameterAuditEntry:
    name: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    finite: bool
    trainable: bool
    lower_bound: float | None
    upper_bound: float | None
    minimum_fraction_of_range: float | None
    maximum_fraction_of_range: float | None
    near_boundary_fraction: float | None


def audit_parameters(
    model: RetinaModel,
    decoder: TiedLocalDecoder,
) -> tuple[ParameterAuditEntry, ...]:
    rgc = model.rgc
    tensors = {
        "rgc.spatial_sigma": (rgc.spatial_sigma, rgc.sigma_bounds),
        "rgc.sustained_mix": (rgc.sustained_mix, (0.0, 1.0)),
        "rgc.membrane_tau_ms": (rgc.membrane_tau_ms, rgc.tau_bounds),
        "rgc.adaptation_tau_ms": (rgc.adaptation_tau_ms, rgc.tau_bounds),
        "rgc.adaptation_gain": (rgc.adaptation_gain, (0.0, rgc.adaptation_gain_max)),
        "rgc.amacrine_gain": (rgc.amacrine_gain, (0.0, rgc.amacrine_gain_max)),
        "rgc.threshold": (rgc.threshold, (0.02, 2.0)),
        "rgc.subunit_tau_ms": (rgc.subunit_tau_ms, rgc.tau_bounds),
        "rgc.subunit_gain": (rgc.subunit_gain, (0.0, rgc.subunit_gain_max)),
        "rgc.readout_rate_tau_ms": (rgc.readout_rate_tau_ms, None),
        "decoder.unit_gain": (decoder.unit_gain, (0.0, decoder.gain_max)),
        "decoder.cone_bias": (decoder.cone_bias, None),
    }
    fixed_names = {"rgc.readout_rate_tau_ms"}
    return tuple(
        _entry(name, tensor, name not in fixed_names, bounds)
        for name, (tensor, bounds) in tensors.items()
    )


def _entry(
    name: str,
    tensor: torch.Tensor,
    trainable: bool,
    bounds: tuple[float, float] | None,
) -> ParameterAuditEntry:
    detached = tensor.detach()
    lower, upper = bounds if bounds is not None else (None, None)
    if bounds is None:
        minimum_fraction = maximum_fraction = near_boundary = None
    else:
        width = max(upper - lower, torch.finfo(detached.dtype).eps)
        fractions = (detached - lower) / width
        minimum_fraction = float(fractions.min())
        maximum_fraction = float(fractions.max())
        near_boundary = float(
            ((fractions <= 0.01) | (fractions >= 0.99)).float().mean()
        )
    return ParameterAuditEntry(
        name=name,
        shape=tuple(detached.shape),
        minimum=float(detached.min()),
        maximum=float(detached.max()),
        finite=bool(torch.isfinite(detached).all()),
        trainable=trainable,
        lower_bound=lower,
        upper_bound=upper,
        minimum_fraction_of_range=minimum_fraction,
        maximum_fraction_of_range=maximum_fraction,
        near_boundary_fraction=near_boundary,
    )


__all__ = ["ParameterAuditEntry", "audit_parameters"]
