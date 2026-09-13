from __future__ import annotations

import hashlib

import torch
from torch import nn

from models.mechanistic_retina.pathway_spatial_geometry import (
    PathwaySpatialGeometry, PathwaySpatialGeometryError, validate_pathway_spatial_geometry,
)
from models.mechanistic_retina.support_partition import SupportPartition, SupportPartitionRequest


def external_supports(geometry: PathwaySpatialGeometry | None, request: SupportPartitionRequest) -> SupportPartition:
    if geometry is None:
        raise PathwaySpatialGeometryError("external geometry must be supplied explicitly")
    for positions in (request.cone_positions, request.cell_positions):
        if positions.ndim != 2 or positions.shape[1] != 2 or not bool(torch.isfinite(positions).all()):
            raise PathwaySpatialGeometryError("external positions must be finite [count,2]")
    for support in (geometry.bc_support, geometry.ac_support):
        if not bool(((support == 0) | (support == 1)).all()):
            raise PathwaySpatialGeometryError("external supports must be binary masks")
    h1 = (torch.cdist(request.cell_positions.float(), request.cone_positions.float()) <= request.h1_radius_deg).float()
    supports = SupportPartition(geometry.bc_support, geometry.ac_support, h1)
    validate_pathway_spatial_geometry(geometry, cell_count=request.cell_positions.shape[0],
                                    cone_count=request.cone_positions.shape[0], spatial_mode_count=2,
                                    expected_supports=supports)
    if not bool(h1.any(dim=1).all()):
        raise PathwaySpatialGeometryError("external geometry requires nonempty H1 support")
    for support in (supports.bc, supports.ac):
        if not bool((geometry.spatial_basis * support[:, None]).sum(-1).gt(0).all()):
            raise PathwaySpatialGeometryError("every spatial mode must have positive supported mass")
    return supports


def geometry_sha256(cones: torch.Tensor, cells: torch.Tensor, geometry: PathwaySpatialGeometry,
                    edges: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("cones", cones), ("cells", cells), ("basis", geometry.spatial_basis),
                        ("bc", geometry.bc_support), ("ac", geometry.ac_support), ("edges", edges)):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def register_external_geometry(model: nn.Module, cones: torch.Tensor, cells: torch.Tensor,
                               geometry: PathwaySpatialGeometry | None, edges: torch.Tensor | None,
                               expected_sha256: str) -> None:
    if geometry is None or edges is None:
        raise PathwaySpatialGeometryError("external contract requires fixed geometry and explicit graph")
    if geometry_sha256(cones, cells, geometry, edges) != expected_sha256:
        raise PathwaySpatialGeometryError("external geometry SHA256 mismatch")
    model.register_buffer("_external_geometry_sha256", torch.tensor(list(expected_sha256.encode()), dtype=torch.uint8))
