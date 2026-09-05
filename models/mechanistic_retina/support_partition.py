from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


_BC_RADIUS: Final = {"midget": 0.06, "parasol": 0.10}
_AC_RADIUS: Final = {"midget": 0.13, "parasol": 0.15}


@dataclass(frozen=True, slots=True)
class SupportPartition:
    bc: torch.Tensor
    ac: torch.Tensor
    h1: torch.Tensor


@dataclass(frozen=True, slots=True)
class SupportPartitionRequest:
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    cell_types: tuple[str, ...]
    h1_radius_deg: float


@dataclass(frozen=True, slots=True)
class SupportPartitionError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def build_support_partition(
    request: SupportPartitionRequest,
) -> SupportPartition:
    if not bool(
        torch.isfinite(request.cone_positions).all()
        and torch.isfinite(request.cell_positions).all()
    ):
        raise SupportPartitionError("Canonical V1 spatial positions must be finite")
    distances = torch.cdist(
        request.cell_positions.float(), request.cone_positions.float()
    )
    bc_radii = distances.new_tensor(
        tuple(_BC_RADIUS[value] for value in request.cell_types)
    )
    ac_radii = distances.new_tensor(
        tuple(_AC_RADIUS[value] for value in request.cell_types)
    )
    bc = distances <= bc_radii[:, None]
    ac = distances <= ac_radii[:, None]
    h1 = distances <= request.h1_radius_deg
    if not bool(bc.any(dim=1).all() and (ac & ~bc).any(dim=1).all() and h1.any(dim=1).all()):
        raise SupportPartitionError(
            "BC and H1 supports must be nonempty; the AC disk must extend beyond BC"
        )
    return SupportPartition(bc.float(), ac.float(), h1.float())


def partition_spatial_basis(
    spatial_basis: torch.Tensor,
    supports: SupportPartition,
) -> torch.Tensor:
    bc = _masked_normalize(spatial_basis, supports.bc)
    ac = _masked_normalize(spatial_basis, supports.ac)
    return torch.stack((bc, bc, ac, ac), dim=1)


def _masked_normalize(basis: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    masked = basis * support[:, None]
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)


__all__ = [
    "SupportPartition",
    "SupportPartitionError",
    "SupportPartitionRequest",
    "build_support_partition",
    "partition_spatial_basis",
]
