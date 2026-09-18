from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
import math

import torch
from torch import Tensor
from torch.nn import functional as F


DT_MS = 1000.0 / 150.0


def regular_pixel_bounds(size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Tessellate the same [-1,1]^2 deg field; never resize stimulus values."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("grid size must be a positive integer")
    edges = torch.linspace(-1.0, 1.0, size + 1, dtype=dtype)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return torch.stack((torch.stack((edges[xx], edges[xx + 1]), -1),
                        torch.stack((edges[yy], edges[yy + 1]), -1)), -2).reshape(-1, 2, 2)


def validate_pixel_bounds(bounds: Tensor) -> None:
    if bounds.ndim != 3 or bounds.shape[1:] != (2, 2) or not bounds.is_floating_point():
        raise ValueError("pixel_bounds must be floating [P,xy,lower_upper]")
    if bounds.shape[0] == 0 or not bool(torch.isfinite(bounds).all()):
        raise ValueError("pixel bounds must be nonempty and finite")
    if not bool((bounds[..., 1] > bounds[..., 0]).all()):
        raise ValueError("pixel area must be positive")
    # G1 accepts Cartesian tessellations, including reordered/nonuniform cells.
    pairs = [torch.unique(bounds[:, axis], dim=0, sorted=True) for axis in range(2)]
    for intervals in pairs:
        if not torch.equal(intervals[1:, 0], intervals[:-1, 1]):
            raise ValueError("pixel intervals overlap or leave an unknown gap")
    if pairs[0].shape[0] * pairs[1].shape[0] != bounds.shape[0]:
        raise ValueError("G1 requires a complete Cartesian pixel grid")
    if torch.unique(bounds.flatten(1), dim=0).shape[0] != bounds.shape[0]:
        raise ValueError("duplicate pixels are not an independent area")


@dataclass(frozen=True)
class StimulusBatch:
    values: Tensor
    pixel_bounds: Tensor
    pixel_area: Tensor
    time_ms: Tensor
    input_valid: Tensor
    sequence_id: tuple[str, ...]
    split_group_id: tuple[str, ...]
    dt_ms: float = DT_MS
    spatial_unit: str = "deg"
    background_relative: float = 1.0
    calibration: str = "synthetic_known_contrast"
    absolute_photon_rate: str = "unknown_not_used"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def validate(self) -> None:
        x = self.values
        if x.ndim != 4 or x.shape[-1] != 1 or min(x.shape) == 0 or not x.is_floating_point():
            raise ValueError("values must be nonempty floating [B,T,P,C=1]")
        validate_pixel_bounds(self.pixel_bounds)
        if self.pixel_bounds.shape[0] != x.shape[2] or self.pixel_area.shape != (x.shape[2],):
            raise ValueError("pixel count/area does not match values")
        for value in (self.pixel_bounds, self.pixel_area, self.time_ms):
            if value.dtype != x.dtype or value.device != x.device:
                raise ValueError("physical coordinates and values must share dtype/device")
        expected_area = (self.pixel_bounds[..., 1] - self.pixel_bounds[..., 0]).prod(-1)
        if not torch.allclose(self.pixel_area, expected_area, rtol=1e-6, atol=0):
            raise ValueError("pixel_area must agree with bounds")
        if self.input_valid.shape != x.shape or self.input_valid.dtype != torch.bool:
            raise ValueError("input_valid must be boolean with the values shape")
        if self.input_valid.device != x.device or not bool(self.input_valid.all()):
            raise ValueError("G1 rejects missing input; unknown pixels cannot become background")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("stimulus must be finite")
        if not math.isclose(self.dt_ms, DT_MS, rel_tol=0, abs_tol=1e-12):
            raise ValueError("G1 has the frozen 150 Hz numerical clock")
        expected_time = (torch.arange(x.shape[1], device=x.device, dtype=x.dtype) + 0.5) * DT_MS
        if self.time_ms.shape != (x.shape[1],) or not torch.allclose(self.time_ms, expected_time):
            raise ValueError("time_ms must name consecutive bin midpoints from sequence onset")
        if len(self.sequence_id) != x.shape[0] or len(self.split_group_id) != x.shape[0]:
            raise ValueError("one provenance identifier is required per sequence")
        if (self.spatial_unit, self.background_relative, self.calibration, self.absolute_photon_rate) != (
            "deg", 1.0, "synthetic_known_contrast", "unknown_not_used"
        ):
            raise ValueError("G1 accepts only the declared synthetic coordinate/calibration contract")


@dataclass(frozen=True)
class CircuitGeometry:
    input_xy: Tensor
    h_xy: Tensor
    bc_xy: Tensor
    ac_xy: Tensor
    rgc_xy: Tensor
    G: Tensor
    A: Tensor
    D: Tensor
    I: Tensor
    spatial_unit: str = "deg"
    geometry_id: str = "retipath_multiscale_v0_fixed"


class Intervention(StrEnum):
    NORMAL = "NORMAL"
    BLOCK_DIRECT_BC_DRIVE = "BLOCK_DIRECT_BC_DRIVE"


@dataclass(frozen=True)
class InterventionSpec:
    kind: Intervention = Intervention.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", Intervention(self.kind))

    def to_record(self) -> dict[str, str | tuple[str, ...]]:
        blocked = self.kind is Intervention.BLOCK_DIRECT_BC_DRIVE
        return {
            "schema_id": "retipath_multiscale_v0_intervention",
            "name": self.kind.value,
            "target_port": "d_E" if blocked else "none",
            "replacement": "d_E=0; uE=0; gE=softplus(b0)=1" if blocked else "none",
            "time_scope": "entire independent sequence from original V0",
            "preserved_nodes": ("h", "s_B", "o_B", "u_A", "a_AC", "j_I", "gI", "q"),
            "recomputed_nodes": ("gE", "V", "m", "z", "ell", "p") if blocked else (),
            "history_condition": "same supplied events; strictly past; no generated-spike feedback",
        }


@dataclass(frozen=True)
class StateBundle:
    inputs: Mapping[str, Tensor]
    states: Mapping[str, Tensor]
    outputs: Mapping[str, Tensor]
    sequence_id: tuple[str, ...]
    split_group_id: tuple[str, ...]
    intervention: InterventionSpec

    def __post_init__(self) -> None:
        for name in ("inputs", "states", "outputs"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class ObservationBatch:
    layer: str
    variable: str
    selector_node_ids: Tensor
    selector_branch_ids: Tensor
    target: Tensor
    valid: Tensor
    time_indices: Tensor
    unit: str
    noise_model: str
    sigma: float | None
    sequence_id: tuple[str, ...]
    split_group_id: tuple[str, ...]


def select_observation(trace: StateBundle, observation: ObservationBatch, *, logits: bool = False) -> Tensor:
    ports = {("HC", "h"): trace.states["h"], ("BC", "o_B"): trace.outputs["o_B"],
             ("RGC", "events"): trace.outputs["ell" if logits else "p"].unsqueeze(-1)}
    key = (observation.layer, observation.variable)
    if key not in ports or (logits and key != ("RGC", "events")):
        raise ValueError("unsupported or semantically mismatched observation port")
    if (observation.sequence_id, observation.split_group_id) != (trace.sequence_id, trace.split_group_id):
        raise ValueError("observation and stimulus provenance do not match")
    value = ports[key]
    for axis, indices in ((1, observation.time_indices), (2, observation.selector_node_ids),
                          (3, observation.selector_branch_ids)):
        if indices.ndim != 1 or indices.numel() == 0 or indices.dtype != torch.long:
            raise ValueError("selectors must be nonempty int64 vectors")
        if indices.device != value.device or bool((indices < 0).any()) or bool((indices >= value.shape[axis]).any()):
            raise ValueError("selector outside the declared observation axis")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("duplicate selectors would reweight an observation")
        value = value.index_select(axis, indices)
    return value


def observation_loss(trace: StateBundle, observation: ObservationBatch) -> Tensor:
    binary = (observation.layer, observation.variable) == ("RGC", "events")
    prediction = select_observation(trace, observation, logits=binary)
    target, valid = observation.target, observation.valid
    if target.shape != prediction.shape or target.dtype != prediction.dtype or target.device != prediction.device:
        raise ValueError("target must match the selected observation shape/dtype/device")
    if valid.shape != target.shape or valid.dtype != torch.bool or valid.device != target.device:
        raise ValueError("observation mask must be boolean and match target")
    if not bool(torch.isfinite(target[valid]).all()) or not bool(valid.flatten(1).any(1).all()):
        raise ValueError("each sequence needs finite valid observations")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if binary:
        if (observation.unit, observation.noise_model, observation.sigma) != ("binary_occupancy", "bernoulli", None):
            raise ValueError("RGC events require Bernoulli occupancy observations")
        if not bool(((target[valid] == 0) | (target[valid] == 1)).all()):
            raise ValueError("Bernoulli targets must be zero or one")
        element_loss = F.softplus(prediction) - safe_target * prediction
    else:
        if (observation.unit, observation.noise_model, observation.sigma) != ("synthetic_effective", "gaussian", 0.03):
            raise ValueError("G1 continuous observations use identity and known sigma=0.03")
        element_loss = 0.5 * ((prediction - safe_target) / 0.03).square()
    masked = torch.where(valid, element_loss, torch.zeros_like(element_loss)).flatten(1)
    return (masked.sum(1) / valid.flatten(1).sum(1)).mean()
