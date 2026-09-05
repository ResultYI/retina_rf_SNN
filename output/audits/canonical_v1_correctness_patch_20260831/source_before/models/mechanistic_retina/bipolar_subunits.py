from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, assert_never

import torch
from torch import nn
from torch.nn import functional as F

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.delay_parameters import (
    ordered_bounded_delay_ms,
    raw_ordered_delay_from_ms,
)
from models.mechanistic_retina.pathway_temporal import pathway_temporal_basis
from models.mechanistic_retina.pathway_spatial_geometry import (
    PathwaySpatialGeometry,
    validate_pathway_spatial_geometry,
)
from models.mechanistic_retina.support_partition import (
    SupportPartition,
    SupportPartitionRequest,
    build_support_partition,
    partition_spatial_basis,
)
from models.mechanistic_retina.temporal_parameters import (
    ordered_bounded_tau_ms,
    raw_ordered_tau_from_ms,
)


_PATH_COUNT: Final = 4
_SPATIAL_MODE_COUNT: Final = 2
_TEMPORAL_MODE_COUNT: Final = 3


@dataclass(frozen=True, slots=True)
class BipolarOutput:
    sustained: torch.Tensor
    transient: torch.Tensor
    on_sustained: torch.Tensor
    on_transient: torch.Tensor
    off_sustained: torch.Tensor
    off_transient: torch.Tensor


@dataclass(frozen=True, slots=True)
class BipolarConfigurationError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class PathFeatureBank(nn.Module):
    def __init__(
        self,
        config: MechanisticRetinaConfig,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
        pathway_spatial_geometry: PathwaySpatialGeometry | None = None,
    ) -> None:
        super().__init__()
        spatial = (
            _spatial_basis(cone_positions, cell_positions, cell_types)
            if pathway_spatial_geometry is None
            else validate_pathway_spatial_geometry(
                pathway_spatial_geometry,
                cell_count=cell_positions.shape[0],
                cone_count=cone_positions.shape[0],
                spatial_mode_count=_SPATIAL_MODE_COUNT,
            )
        )
        initial_tau = torch.tensor(
            config.bc_basis_tau_ms,
            dtype=torch.float32,
        )
        path_bounds = config.bc_basis_tau_bounds_ms
        tau_bounds = torch.tensor(
            tuple(tuple(bound for _ in range(3)) for bound in path_bounds),
            dtype=torch.float32,
        )
        signs = torch.tensor(
            [_polarity_sign(value) for value in polarities], dtype=torch.float32
        )
        if pathway_spatial_geometry is None:
            supports = build_support_partition(
                SupportPartitionRequest(
                    cone_positions,
                    cell_positions,
                    cell_types,
                    config.h1_radius_deg,
                )
            )
        else:
            h1 = torch.cdist(cell_positions.float(), cone_positions.float())
            h1 = (h1 <= config.h1_radius_deg).to(spatial)
            supports = SupportPartition(
                pathway_spatial_geometry.bc_support.to(spatial),
                pathway_spatial_geometry.ac_support.to(spatial),
                h1,
            )
        mode = ArchitectureMode(config.architecture_mode)
        match mode:
            case ArchitectureMode.MECHANISM_IDENTIFIABLE:
                path_spatial = partition_spatial_basis(spatial, supports)
            case ArchitectureMode.LEGACY:
                path_spatial = spatial[:, None].expand(-1, _PATH_COUNT, -1, -1).clone()
            case unreachable:
                assert_never(unreachable)
        self.register_buffer("spatial_basis", spatial)
        self.register_buffer("path_spatial_basis", path_spatial)
        self.raw_tau = nn.Parameter(raw_ordered_tau_from_ms(initial_tau, tau_bounds))
        self.register_buffer("tau_bounds_ms", tau_bounds)
        initial_delay = torch.tensor(
            config.bc_delay_ms, dtype=torch.float32
        )
        delay_bounds = torch.tensor(
            config.bc_delay_bounds_ms,
            dtype=torch.float32,
        )
        self.raw_delay = nn.Parameter(raw_ordered_delay_from_ms(initial_delay, delay_bounds))
        self.register_buffer("delay_bounds_ms", delay_bounds)
        self.register_buffer("polarity_sign", signs)
        self.register_buffer("bc_support", supports.bc)
        self.register_buffer("ac_support", supports.ac)
        self.register_buffer("h1_support", supports.h1)
        self._lag_steps = config.lag_steps
        self._dt_ms = config.dt_ms

    @property
    def tau_ms(self) -> torch.Tensor:
        return ordered_bounded_tau_ms(self.raw_tau, self.tau_bounds_ms)

    @property
    def temporal_basis(self) -> torch.Tensor:
        return pathway_temporal_basis(
            self.tau_ms,
            self.delay_ms,
            self._lag_steps,
            self._dt_ms,
        )

    @property
    def delay_ms(self) -> torch.Tensor:
        return ordered_bounded_delay_ms(self.raw_delay, self.delay_bounds_ms)

    @property
    def basis_count(self) -> int:
        return _SPATIAL_MODE_COUNT * _TEMPORAL_MODE_COUNT

    @property
    def supports(self) -> SupportPartition:
        return SupportPartition(self.bc_support, self.ac_support, self.h1_support)

    def basis_kernels(self) -> torch.Tensor:
        return torch.einsum(
            "n,prl,npsc->npsrlc",
            self.polarity_sign,
            self.temporal_basis.repeat(2, 1, 1),
            self.path_spatial_basis,
        )

    def forward(self, cones: torch.Tensor) -> torch.Tensor:
        lag_count = self.temporal_basis.shape[-1]
        padded = F.pad(cones, (0, 0, lag_count - 1, 0))
        lagged = padded.unfold(1, lag_count, 1).permute(0, 1, 3, 2)
        return torch.einsum("btlc,npsrlc->btnpsr", lagged, self.basis_kernels())


class BipolarSubunits(nn.Module):
    def __init__(
        self,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
        *,
        shared: bool,
    ) -> None:
        super().__init__()
        group_index, group_count = _group_indices(cell_types, polarities, shared=shared)
        initial = math.log(math.expm1(0.03))
        self.raw_weights = nn.Parameter(
            torch.full((group_count, 2, 2, 3), initial)
            + 0.01 * torch.randn(group_count, 2, 2, 3)
        )
        on = torch.tensor([value == "ON" for value in polarities])
        self.register_buffer("group_index", group_index)
        self.register_buffer("on_mask", on)
        self.register_buffer("off_mask", ~on)

    def positive_weights(self) -> torch.Tensor:
        normalized = F.softmax(self.raw_weights.flatten(1), dim=1).reshape_as(self.raw_weights)
        return normalized[self.group_index]

    def forward(
        self,
        features: torch.Tensor,
        modulation: torch.Tensor,
    ) -> BipolarOutput:
        weights = self.positive_weights()[None, None] * modulation
        states = (features * weights).sum(dim=(-1, -2))
        sustained, transient = states.unbind(-1)
        on = self.on_mask.view(1, 1, -1)
        off = self.off_mask.view(1, 1, -1)
        return BipolarOutput(
            sustained,
            transient,
            sustained * on,
            transient * on,
            sustained * off,
            transient * off,
        )

    def base_kernels(self, basis_kernels: torch.Tensor) -> torch.Tensor:
        basis = basis_kernels[:, :2]
        return (basis * self.positive_weights()[:, :, :, :, None, None]).sum(dim=(2, 3))


def _group_indices(
    cell_types: tuple[str, ...],
    polarities: tuple[str, ...],
    *,
    shared: bool,
) -> tuple[torch.Tensor, int]:
    if not shared:
        return torch.arange(len(cell_types)), len(cell_types)
    keys = tuple(zip(cell_types, polarities, strict=True))
    groups = tuple(dict.fromkeys(keys))
    return torch.tensor(tuple(groups.index(key) for key in keys)), len(groups)


def _spatial_basis(
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
    cell_types: tuple[str, ...],
) -> torch.Tensor:
    sigma_by_type: Final = {
        "midget": (0.05, 0.14),
        "parasol": (0.09, 0.20),
    }
    distances = torch.cdist(cell_positions.float(), cone_positions.float())
    rows = []
    for index, cell_type in enumerate(cell_types):
        sigmas = sigma_by_type[cell_type]
        modes = torch.stack(
            tuple(torch.exp(-0.5 * (distances[index] / sigma) ** 2) for sigma in sigmas)
        )
        rows.append(modes / modes.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.stack(rows)


def _polarity_sign(value: str) -> float:
    signs = {"ON": 1.0, "OFF": -1.0}
    try:
        return signs[value]
    except KeyError as error:
        raise BipolarConfigurationError(f"unsupported polarity: {value}") from error


__all__ = ["BipolarConfigurationError", "BipolarOutput", "BipolarSubunits", "PathwaySpatialGeometry", "PathFeatureBank"]
