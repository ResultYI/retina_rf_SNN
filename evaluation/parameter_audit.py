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


def audit_parameters(
    model: RetinaModel,
    decoder: TiedLocalDecoder,
) -> tuple[ParameterAuditEntry, ...]:
    rgc = model.rgc
    tensors = {
        "rgc.spatial_sigma": rgc.spatial_sigma,
        "rgc.sustained_mix": rgc.sustained_mix,
        "rgc.membrane_tau_ms": rgc.membrane_tau_ms,
        "rgc.adaptation_tau_ms": rgc.adaptation_tau_ms,
        "rgc.adaptation_gain": rgc.adaptation_gain,
        "rgc.amacrine_gain": rgc.amacrine_gain,
        "rgc.threshold": rgc.threshold,
        "rgc.subunit_tau_ms": rgc.subunit_tau_ms,
        "rgc.subunit_gain": rgc.subunit_gain,
        "rgc.readout_rate_tau_ms": rgc.readout_rate_tau_ms,
        "decoder.unit_gain": decoder.unit_gain,
        "decoder.cone_bias": decoder.cone_bias,
    }
    fixed_names = {"rgc.readout_rate_tau_ms"}
    return tuple(
        _entry(name, tensor, name not in fixed_names)
        for name, tensor in tensors.items()
    )


def _entry(
    name: str,
    tensor: torch.Tensor,
    trainable: bool,
) -> ParameterAuditEntry:
    detached = tensor.detach()
    return ParameterAuditEntry(
        name=name,
        shape=tuple(detached.shape),
        minimum=float(detached.min()),
        maximum=float(detached.max()),
        finite=bool(torch.isfinite(detached).all()),
        trainable=trainable,
    )


__all__ = ["ParameterAuditEntry", "audit_parameters"]
