from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from evaluation.mechanistic_retina.karamanlis_rf_centers import (
    _central_contour,
    _gaussian_blur,
    _peak_component,
    _pixel_axis,
)


RF_CONTOUR_FRACTION: Final = 0.25
RF_BLUR_SIGMA_SCREEN_PX: Final = 4.0
SELF_CONNECTION_INITIAL: Final = 1.0
NEIGHBOR_CONNECTION_INITIAL: Final = 0.1


@dataclass(frozen=True, slots=True)
class RFLocalityError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class RFMapGrid:
    x_centers_px: np.ndarray
    y_centers_px: np.ndarray
    stixel_width_px: int
    stixel_height_px: int
    screen_pixel_size_um: float
    origin_px: np.ndarray

    def __post_init__(self) -> None:
        valid_axes = (
            self.x_centers_px.ndim == 1
            and self.y_centers_px.ndim == 1
            and self.x_centers_px.size > 0
            and self.y_centers_px.size > 0
            and np.isfinite(self.x_centers_px).all()
            and np.isfinite(self.y_centers_px).all()
        )
        valid_projection = (
            self.stixel_width_px > 0
            and self.stixel_height_px > 0
            and np.isfinite(self.screen_pixel_size_um)
            and self.screen_pixel_size_um > 0
            and self.origin_px.shape == (2,)
            and np.isfinite(self.origin_px).all()
        )
        if not valid_axes or not valid_projection:
            raise RFLocalityError("RF map grid requires finite experimental projection geometry")


@dataclass(frozen=True, slots=True)
class RFSpatialExtent:
    support_mask: np.ndarray
    contour_um: np.ndarray
    center_um: np.ndarray
    area_um2: float
    equivalent_radius_um: float
    width_um: float
    height_um: float
    touches_boundary: bool = False


@dataclass(frozen=True, slots=True)
class RFLocalityCell:
    original_index: int
    cell_id: str
    cell_type: str
    polarity: str
    extent: RFSpatialExtent


@dataclass(frozen=True, slots=True)
class RFLocalityGraph:
    cells: tuple[RFLocalityCell, ...]
    adjacency: np.ndarray
    overlap_adjacency: np.ndarray
    proximity_adjacency: np.ndarray
    center_distance_um: np.ndarray
    edge_index: np.ndarray
    raw_connections: np.ndarray


def extract_rf_spatial_extent(
    spatial_rf: np.ndarray,
    grid: RFMapGrid,
) -> RFSpatialExtent:
    expected_shape = (grid.y_centers_px.size, grid.x_centers_px.size)
    if spatial_rf.shape != expected_shape or not np.isfinite(spatial_rf).all():
        raise RFLocalityError("spatial RF must match the finite experimental grid")
    upsampled = np.repeat(
        np.repeat(spatial_rf, grid.stixel_height_px, axis=0),
        grid.stixel_width_px,
        axis=1,
    )
    smoothed = _gaussian_blur(upsampled, RF_BLUR_SIGMA_SCREEN_PX)
    peak = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    peak_value = float(smoothed[peak])
    if not np.isfinite(peak_value) or peak_value <= 0:
        raise RFLocalityError("spatial RF lacks a finite positive central peak")
    support = _peak_component(smoothed >= RF_CONTOUR_FRACTION * peak_value, peak)
    contour_result = _central_contour(smoothed)
    if contour_result is None:
        raise RFLocalityError("spatial RF 25-percent central contour is unavailable")
    contour, touches_boundary = contour_result
    x_screen_px = _pixel_axis(grid.x_centers_px, grid.stixel_width_px)
    y_screen_px = _pixel_axis(grid.y_centers_px, grid.stixel_height_px)
    contour_screen_px = np.column_stack(
        (x_screen_px[contour[:, 1]], y_screen_px[contour[:, 0]])
    )
    contour_um = (
        contour_screen_px - grid.origin_px[None]
    ) * grid.screen_pixel_size_um
    center_um = np.median(contour_um, axis=0)
    area_um2 = float(support.sum()) * grid.screen_pixel_size_um**2
    equivalent_radius_um = float(np.sqrt(area_um2 / np.pi))
    width_um = float(np.ptp(contour_um[:, 0]) + grid.screen_pixel_size_um)
    height_um = float(np.ptp(contour_um[:, 1]) + grid.screen_pixel_size_um)
    return RFSpatialExtent(
        support_mask=support,
        contour_um=np.asarray(contour_um, dtype=np.float64),
        center_um=np.asarray(center_um, dtype=np.float64),
        area_um2=area_um2,
        equivalent_radius_um=equivalent_radius_um,
        width_um=width_um,
        height_um=height_um,
        touches_boundary=touches_boundary,
    )


def build_rf_locality_graph(
    cells: tuple[RFLocalityCell, ...],
) -> RFLocalityGraph:
    if not cells:
        raise RFLocalityError("locality graph requires at least one RF cell")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise RFLocalityError("locality graph cell IDs must be unique")
    support_shape = cells[0].extent.support_mask.shape
    for cell in cells:
        valid_extent = (
            cell.extent.support_mask.shape == support_shape
            and cell.extent.support_mask.dtype == np.bool_
            and cell.extent.contour_um.ndim == 2
            and cell.extent.contour_um.shape[1] == 2
            and cell.extent.center_um.shape == (2,)
            and np.isfinite(cell.extent.contour_um).all()
            and np.isfinite(cell.extent.center_um).all()
            and np.isfinite(cell.extent.equivalent_radius_um)
            and cell.extent.equivalent_radius_um > 0
        )
        if not valid_extent:
            raise RFLocalityError("locality graph contains an invalid RF extent")
    centers = np.stack(tuple(cell.extent.center_um for cell in cells))
    radii = np.asarray(
        tuple(cell.extent.equivalent_radius_um for cell in cells), dtype=np.float64
    )
    center_distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    same_group = np.asarray(
        [
            [
                target.cell_type == source.cell_type
                and target.polarity == source.polarity
                for source in cells
            ]
            for target in cells
        ],
        dtype=bool,
    )
    overlap = np.asarray(
        [
            [
                bool(np.any(target.extent.support_mask & source.extent.support_mask))
                for source in cells
            ]
            for target in cells
        ],
        dtype=bool,
    )
    overlap &= same_group
    proximity = same_group & (center_distance <= radii[:, None] + radii[None])
    adjacency = overlap | proximity
    np.fill_diagonal(adjacency, True)
    edge_index = np.argwhere(adjacency).T.astype(np.int64, copy=False)
    positive = np.where(
        edge_index[0] == edge_index[1],
        SELF_CONNECTION_INITIAL,
        NEIGHBOR_CONNECTION_INITIAL,
    ).astype(np.float32)
    raw_connections = np.log(np.expm1(positive)).astype(np.float32)
    return RFLocalityGraph(
        cells=cells,
        adjacency=adjacency,
        overlap_adjacency=overlap,
        proximity_adjacency=proximity,
        center_distance_um=center_distance,
        edge_index=edge_index,
        raw_connections=raw_connections,
    )


__all__ = [
    "NEIGHBOR_CONNECTION_INITIAL",
    "RFLocalityCell",
    "RFLocalityError",
    "RFLocalityGraph",
    "RFMapGrid",
    "RFSpatialExtent",
    "SELF_CONNECTION_INITIAL",
    "build_rf_locality_graph",
    "extract_rf_spatial_extent",
]
