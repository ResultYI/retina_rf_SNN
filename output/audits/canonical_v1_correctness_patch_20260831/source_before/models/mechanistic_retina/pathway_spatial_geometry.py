from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PathwaySpatialGeometry:
    spatial_basis: torch.Tensor
    bc_support: torch.Tensor
    ac_support: torch.Tensor


@dataclass(frozen=True, slots=True)
class PathwaySpatialGeometryError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def validate_pathway_spatial_geometry(
    geometry: PathwaySpatialGeometry,
    *,
    cell_count: int,
    cone_count: int,
    spatial_mode_count: int,
) -> torch.Tensor:
    expected_basis = (cell_count, spatial_mode_count, cone_count)
    expected_support = (cell_count, cone_count)
    if geometry.spatial_basis.shape != expected_basis:
        raise PathwaySpatialGeometryError(
            "RF-derived spatial basis must be [cell,2,cone]"
        )
    if (
        geometry.bc_support.shape != expected_support
        or geometry.ac_support.shape != expected_support
    ):
        raise PathwaySpatialGeometryError(
            "RF-derived BC/AC supports must be [cell,cone]"
        )
    tensors = (
        geometry.spatial_basis,
        geometry.bc_support,
        geometry.ac_support,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise PathwaySpatialGeometryError(
            "RF-derived spatial geometry must be finite"
        )
    bc = geometry.bc_support > 0
    ac = geometry.ac_support > 0
    if (
        bool((bc & ~ac).any())
        or not bool(bc.any(dim=1).all())
        or not bool((ac & ~bc).any(dim=1).all())
    ):
        raise PathwaySpatialGeometryError(
            "RF-derived BC support must be a nonempty strict subset of AC support"
        )
    if bool((geometry.spatial_basis < 0).any()):
        raise PathwaySpatialGeometryError(
            "RF-derived spatial basis must be nonnegative"
        )
    return geometry.spatial_basis


__all__ = ["PathwaySpatialGeometry", "PathwaySpatialGeometryError", "validate_pathway_spatial_geometry"]
