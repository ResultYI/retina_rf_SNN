from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, assert_never

import torch
from torch import nn

from data.rgc_response import CellMetadata


ParameterRole: TypeAlias = Literal[
    "rgc_type_base",
    "rgc_cell_residual",
    "response_bias",
    "synaptic_gain",
    "polarity_pathway",
    "other",
]


@dataclass(frozen=True, slots=True)
class ParameterAuditContext:
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]

    @classmethod
    def from_cells(cls, cells: CellMetadata) -> ParameterAuditContext:
        return cls(tuple(sorted(set(cells.type_ids))), cells.ids)


@dataclass(frozen=True, slots=True)
class ParameterDelta:
    group: str
    role: ParameterRole
    name: str
    absolute_norm: float
    relative_norm: float | None
    element_labels: tuple[str, ...]
    delta_values: tuple[float, ...]
    trainable: bool


@dataclass(frozen=True, slots=True)
class TypeDifferential:
    separation_ratio: float
    opposition_cosine: float


@dataclass(frozen=True, slots=True)
class ResponseReadoutAudit:
    initial_response_bias: tuple[float, ...]
    calibrated_response_bias: tuple[float, ...]
    trained_response_bias: tuple[float, ...]
    initial_effective_synaptic_gain: tuple[float, ...]
    calibrated_effective_synaptic_gain: tuple[float, ...]
    trained_effective_synaptic_gain: tuple[float, ...]
    initial_bipolar_readout_gain: tuple[tuple[float, ...], ...] = ()
    calibrated_bipolar_readout_gain: tuple[tuple[float, ...], ...] = ()
    trained_bipolar_readout_gain: tuple[tuple[float, ...], ...] = ()
    initial_amacrine_readout_gain: tuple[tuple[float, ...], ...] = ()
    calibrated_amacrine_readout_gain: tuple[tuple[float, ...], ...] = ()
    trained_amacrine_readout_gain: tuple[tuple[float, ...], ...] = ()


def audit_parameter_deltas(
    trained: nn.Module,
    initialized: nn.Module,
    context: ParameterAuditContext | None = None,
) -> tuple[ParameterDelta, ...]:
    initialized_parameters = dict(initialized.named_parameters())
    values = []
    for name, trained_parameter in trained.named_parameters():
        initialized_parameter = initialized_parameters[name]
        delta = trained_parameter.detach() - initialized_parameter.detach()
        delta_norm = float(torch.linalg.vector_norm(delta))
        initialized_norm = float(
            torch.linalg.vector_norm(initialized_parameter.detach())
        )
        role = parameter_role(name)
        match role:
            case "rgc_type_base":
                labels = _rgc_type_base_labels(trained, context)
            case "rgc_cell_residual":
                labels = () if context is None else context.cell_ids
            case "response_bias" | "synaptic_gain":
                labels = () if context is None else context.cell_ids
            case "polarity_pathway":
                labels = ("ON", "OFF")
            case "other":
                labels = ()
            case unreachable:
                assert_never(unreachable)
        values.append(
            ParameterDelta(
                group=name.partition(".")[0],
                role=role,
                name=name,
                absolute_norm=delta_norm,
                relative_norm=(
                    None if initialized_norm == 0 else delta_norm / initialized_norm
                ),
                element_labels=labels,
                delta_values=tuple(float(value) for value in delta.flatten()),
                trainable=trained_parameter.requires_grad,
            )
        )
    return tuple(values)


def parameter_role(name: str) -> ParameterRole:
    if name == "rgc.response_bias":
        return "response_bias"
    if name == "rgc.synaptic_gain_raw":
        return "synaptic_gain"
    if name.startswith("rgc.") and name.endswith(".type_base_raw"):
        return "rgc_type_base"
    if name.startswith("rgc.") and name.endswith(".cell_residual_raw"):
        return "rgc_cell_residual"
    if name in {
        "bipolar.raw_polarity_gain",
        "bipolar.raw_polarity_threshold",
    }:
        return "polarity_pathway"
    return "other"


def audit_response_readout(
    trained: nn.Module,
    calibrated: nn.Module,
) -> ResponseReadoutAudit:
    calibrated_bias = calibrated.rgc.response_bias.detach().cpu()
    trained_bias = trained.rgc.response_bias.detach().cpu()
    calibrated_gain = calibrated.rgc.synaptic_gain().detach().cpu()
    trained_gain = trained.rgc.synaptic_gain().detach().cpu()
    calibrated_bipolar = _readout_gain(calibrated.rgc, "bipolar_readout_gain")
    trained_bipolar = _readout_gain(trained.rgc, "bipolar_readout_gain")
    calibrated_amacrine = _readout_gain(calibrated.rgc, "amacrine_readout_gain")
    trained_amacrine = _readout_gain(trained.rgc, "amacrine_readout_gain")
    return ResponseReadoutAudit(
        initial_response_bias=tuple(0.0 for _ in calibrated_bias),
        calibrated_response_bias=tuple(float(value) for value in calibrated_bias),
        trained_response_bias=tuple(float(value) for value in trained_bias),
        initial_effective_synaptic_gain=tuple(1.0 for _ in calibrated_gain),
        calibrated_effective_synaptic_gain=tuple(
            float(value) for value in calibrated_gain
        ),
        trained_effective_synaptic_gain=tuple(float(value) for value in trained_gain),
        initial_bipolar_readout_gain=_zero_matrix(calibrated_bipolar),
        calibrated_bipolar_readout_gain=calibrated_bipolar,
        trained_bipolar_readout_gain=trained_bipolar,
        initial_amacrine_readout_gain=_zero_matrix(calibrated_amacrine),
        calibrated_amacrine_readout_gain=calibrated_amacrine,
        trained_amacrine_readout_gain=trained_amacrine,
    )


def _readout_gain(
    rgc: nn.Module,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    value = getattr(rgc, name, None)
    if value is None:
        return ()
    return tuple(
        tuple(float(item) for item in row)
        for row in value.detach().cpu()
    )


def _zero_matrix(
    values: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(0.0 for _ in row) for row in values)


def _rgc_type_base_labels(
    trained: nn.Module,
    context: ParameterAuditContext | None,
) -> tuple[str, ...]:
    rgc = getattr(trained, "rgc", None)
    labels = () if rgc is None else getattr(rgc, "parameter_group_labels", ())
    if labels:
        return tuple(str(label) for label in labels)
    return () if context is None else context.type_ids


def type_differential(
    midget: torch.Tensor,
    parasol: torch.Tensor,
) -> TypeDifferential:
    midget_vector = midget.flatten()
    parasol_vector = parasol.flatten()
    epsilon = torch.finfo(midget_vector.dtype).eps
    midget_norm = torch.linalg.vector_norm(midget_vector)
    parasol_norm = torch.linalg.vector_norm(parasol_vector)
    separation = torch.linalg.vector_norm(parasol_vector - midget_vector) / (
        midget_norm + parasol_norm + epsilon
    )
    opposition = -torch.dot(midget_vector, parasol_vector) / (
        midget_norm * parasol_norm + epsilon
    )
    return TypeDifferential(float(separation), float(opposition))


__all__ = [
    "ParameterAuditContext",
    "ParameterDelta",
    "ParameterRole",
    "ResponseReadoutAudit",
    "TypeDifferential",
    "audit_parameter_deltas",
    "audit_response_readout",
    "parameter_role",
    "type_differential",
]
