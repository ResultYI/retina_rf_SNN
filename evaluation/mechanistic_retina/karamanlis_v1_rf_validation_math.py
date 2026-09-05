from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class RFValidationMathError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StixelProjection:
    indices: np.ndarray
    weights: np.ndarray
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class SeparableRF:
    spatial: torch.Tensor
    temporal: torch.Tensor


def build_stixel_projection(
    cone_blocks_screen_indices: np.ndarray,
    *,
    x_centers_px: np.ndarray,
    y_centers_px: np.ndarray,
    stixel_width_px: int,
    stixel_height_px: int,
) -> StixelProjection:
    blocks = np.asarray(cone_blocks_screen_indices, dtype=np.int64)
    if blocks.ndim != 2 or blocks.shape[1] != 4:
        raise RFValidationMathError("cone blocks must be [cone,4]")
    x_starts = _stixel_starts(x_centers_px, stixel_width_px)
    y_starts = _stixel_starts(y_centers_px, stixel_height_px)
    rows: list[list[tuple[int, float]]] = []
    for y0, y1, x0, x1 in blocks:
        area = int(y1 - y0) * int(x1 - x0)
        if area <= 0:
            raise RFValidationMathError("cone blocks must have positive area")
        overlaps: list[tuple[int, float]] = []
        for row in _overlapping_axis(y0, y1, y_starts, stixel_height_px):
            y_overlap = _overlap(y0, y1, y_starts[row], stixel_height_px)
            for column in _overlapping_axis(x0, x1, x_starts, stixel_width_px):
                x_overlap = _overlap(x0, x1, x_starts[column], stixel_width_px)
                overlaps.append(
                    (row * x_starts.size + column, x_overlap * y_overlap / area)
                )
        if not overlaps or not np.isclose(sum(weight for _, weight in overlaps), 1.0):
            raise RFValidationMathError("cone block is not covered by the STA grid")
        rows.append(overlaps)
    width = max(len(row) for row in rows)
    indices = np.full((len(rows), width), -1, dtype=np.int64)
    weights = np.zeros((len(rows), width), dtype=np.float32)
    for row_index, overlaps in enumerate(rows):
        for slot, (index, weight) in enumerate(overlaps):
            indices[row_index, slot] = index
            weights[row_index, slot] = weight
    return StixelProjection(indices, weights, y_starts.size, x_starts.size)


def project_cone_rf_to_stixels(
    cone_rf: torch.Tensor,
    projection: StixelProjection,
) -> torch.Tensor:
    if cone_rf.shape[-1] != projection.indices.shape[0]:
        raise RFValidationMathError("cone RF and projection cone counts differ")
    output = cone_rf.new_zeros(
        (*cone_rf.shape[:-1], projection.height * projection.width)
    )
    indices = torch.as_tensor(projection.indices, device=cone_rf.device)
    weights = torch.as_tensor(
        projection.weights, device=cone_rf.device, dtype=cone_rf.dtype
    )
    for slot in range(indices.shape[1]):
        valid = indices[:, slot] >= 0
        output.index_add_(
            -1,
            indices[valid, slot],
            cone_rf[..., valid] * weights[valid, slot],
        )
    return output.reshape(*cone_rf.shape[:-1], projection.height, projection.width)


def separable_projection(stixel_rf: torch.Tensor) -> SeparableRF:
    if stixel_rf.ndim < 3:
        raise RFValidationMathError("RF must be cell x lag x spatial dimensions")
    flattened = stixel_rf.flatten(start_dim=2)
    temporal = flattened.sum(dim=-1)
    energy = temporal.square().sum(dim=-1)
    if not bool(torch.isfinite(flattened).all()) or bool((energy <= 1e-16).any()):
        raise RFValidationMathError("RF temporal projection is non-finite or zero")
    spatial = torch.einsum("cl,cls->cs", temporal, flattened) / energy[:, None]
    flip = spatial.amin(dim=-1).abs() > spatial.amax(dim=-1)
    spatial = torch.where(flip[:, None], -spatial, spatial)
    temporal = torch.where(flip[:, None], -temporal, temporal)
    return SeparableRF(
        spatial.reshape(stixel_rf.shape[0], *stixel_rf.shape[2:]), temporal
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    return (
        float(np.dot(left_values, right_values) / denominator)
        if denominator > 0
        else float("nan")
    )


def _stixel_starts(centers: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64)
    starts = values - (size + 1) / 2
    if size < 1 or values.ndim != 1 or not np.allclose(starts, np.rint(starts)):
        raise RFValidationMathError("STA stixels must have integer screen-pixel bounds")
    return np.rint(starts).astype(np.int64)


def _overlapping_axis(
    start: int,
    stop: int,
    stixel_starts: np.ndarray,
    stixel_size: int,
) -> np.ndarray:
    return np.flatnonzero(
        (stixel_starts < stop) & (stixel_starts + stixel_size > start)
    )


def _overlap(start: int, stop: int, stixel_start: int, stixel_size: int) -> int:
    return max(0, min(stop, stixel_start + stixel_size) - max(start, stixel_start))


__all__ = [
    "RFValidationMathError",
    "SeparableRF",
    "StixelProjection",
    "build_stixel_projection",
    "cosine",
    "project_cone_rf_to_stixels",
    "separable_projection",
]
