from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from evaluation.mechanistic_retina.gollisch_white_noise import (
    recreate_binary_white_noise,
)


@dataclass(frozen=True, slots=True)
class RFGrid:
    x_centers_px: np.ndarray
    y_centers_px: np.ndarray
    stixel_width_px: int
    stixel_height_px: int


@dataclass(frozen=True, slots=True)
class RFSuccess:
    center_x_px: float
    center_y_px: float
    spatial_rf: np.ndarray
    temporal_filter: np.ndarray
    robust_sd: float
    significant_pixel_count: int
    contour_point_count: int
    touches_boundary: bool


@dataclass(frozen=True, slots=True)
class RFFailure:
    reason: str


RFEstimate = RFSuccess | RFFailure


@dataclass(frozen=True, slots=True)
class ConditionResult:
    centers_px: np.ndarray
    spatial_rfs: np.ndarray
    temporal_filters: np.ndarray
    robust_sds: np.ndarray
    significant_counts: np.ndarray
    contour_counts: np.ndarray
    touches_boundary: np.ndarray
    failure_reasons: tuple[str | None, ...]


def batch_spike_triggered_average(
    stimulus: np.ndarray,
    responses: np.ndarray,
    lag_count: int,
) -> np.ndarray:
    if stimulus.ndim != 3 or responses.ndim != 2:
        raise ValueError("stimulus must be T×Y×X and responses T×cells")
    if stimulus.shape[0] != responses.shape[0]:
        raise ValueError("stimulus and responses must share the time axis")
    if lag_count < 1 or lag_count > stimulus.shape[0]:
        raise ValueError("lag_count must fit within the stimulus")
    time_count, height, width = stimulus.shape
    centered = stimulus.astype(np.float32) - stimulus.mean(axis=0, dtype=np.float64)
    flat = centered.reshape(time_count, height * width)
    response_values = responses.astype(np.float32, copy=False)
    output = np.zeros(
        (responses.shape[1], lag_count, height * width), dtype=np.float32
    )
    for lag in range(lag_count):
        weights = response_values[lag:]
        denominators = weights.sum(axis=0)
        nonzero = denominators > 0
        if np.any(nonzero):
            products = weights[:, nonzero].T @ flat[: time_count - lag]
            output[nonzero, lag] = products / denominators[nonzero, None]
    return output.reshape(responses.shape[1], lag_count, height, width)


def spike_triggered_average(
    stimulus: np.ndarray,
    response: np.ndarray,
    lag_count: int,
) -> np.ndarray:
    if response.ndim != 1:
        raise ValueError("response must be one-dimensional")
    return batch_spike_triggered_average(stimulus, response[:, None], lag_count)[0]


def _robust_sd(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _pixel_axis(centers: np.ndarray, stixel_size: int) -> np.ndarray:
    if stixel_size % 2 != 1:
        raise ValueError("stixel dimensions must be odd")
    start = float(centers[0]) - (stixel_size - 1) / 2
    return start + np.arange(centers.size * stixel_size, dtype=np.float64)


def _gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    horizontal = np.pad(values, ((0, 0), (radius, radius)), mode="reflect")
    horizontal_windows = np.lib.stride_tricks.sliding_window_view(
        horizontal, kernel.size, axis=1
    )
    horizontal_blur = np.einsum("yxk,k->yx", horizontal_windows, kernel)
    vertical = np.pad(horizontal_blur, ((radius, radius), (0, 0)), mode="reflect")
    vertical_windows = np.lib.stride_tricks.sliding_window_view(
        vertical, kernel.size, axis=0
    )
    return np.einsum("yxk,k->yx", vertical_windows, kernel)


def _peak_component(mask: np.ndarray, peak: tuple[int, int]) -> np.ndarray:
    component = np.zeros_like(mask, dtype=bool)
    pending = deque([peak])
    component[peak] = True
    while pending:
        row, column = pending.popleft()
        for row_step in (-1, 0, 1):
            for column_step in (-1, 0, 1):
                neighbor = row + row_step, column + column_step
                if (
                    0 <= neighbor[0] < mask.shape[0]
                    and 0 <= neighbor[1] < mask.shape[1]
                    and mask[neighbor]
                    and not component[neighbor]
                ):
                    component[neighbor] = True
                    pending.append(neighbor)
    return component


def _central_contour(spatial_rf: np.ndarray) -> tuple[np.ndarray, bool] | None:
    peak = np.unravel_index(int(np.argmax(spatial_rf)), spatial_rf.shape)
    peak_value = float(spatial_rf[peak])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return None
    mask = spatial_rf >= 0.25 * peak_value
    component = _peak_component(mask, peak)
    touches_boundary = bool(
        component[0].any()
        or component[-1].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )
    padded = np.pad(component, 1, constant_values=False)
    interior = component.copy()
    for row_step in (-1, 0, 1):
        for column_step in (-1, 0, 1):
            interior &= padded[
                1 + row_step : 1 + row_step + component.shape[0],
                1 + column_step : 1 + column_step + component.shape[1],
            ]
    contour = np.argwhere(component & ~interior)
    if contour.shape[0] < 3:
        return None
    return contour, touches_boundary


def estimate_rf(sta: np.ndarray, grid: RFGrid) -> RFEstimate:
    if sta.ndim != 3:
        raise ValueError("STA must be lag×Y×X")
    if sta.shape[1:] != (grid.y_centers_px.size, grid.x_centers_px.size):
        raise ValueError("STA spatial shape does not match RF grid")
    robust_sd = _robust_sd(sta)
    if not np.isfinite(robust_sd) or robust_sd <= 0:
        return RFFailure("non-positive STA robust SD")
    significant = np.max(np.abs(sta), axis=0) > 4.5 * robust_sd
    significant_count = int(significant.sum())
    if significant_count == 0:
        return RFFailure("no STA pixels exceed 4.5 robust SD")
    temporal_filter = sta[:, significant].mean(axis=1)
    temporal_energy = float(np.dot(temporal_filter, temporal_filter))
    if not np.isfinite(temporal_energy) or temporal_energy <= 0:
        return RFFailure("temporal filter has no finite energy")
    spatial_rf = np.tensordot(temporal_filter, sta, axes=(0, 0)) / temporal_energy
    if abs(float(spatial_rf.min())) > float(spatial_rf.max()):
        spatial_rf = -spatial_rf
        temporal_filter = -temporal_filter
    upsampled = np.repeat(
        np.repeat(spatial_rf, grid.stixel_height_px, axis=0),
        grid.stixel_width_px,
        axis=1,
    )
    smoothed = _gaussian_blur(upsampled, sigma=4.0)
    contour_result = _central_contour(smoothed)
    if contour_result is None:
        return RFFailure("25% RF contour could not be formed")
    contour, touches_boundary = contour_result
    x_pixels = _pixel_axis(grid.x_centers_px, grid.stixel_width_px)
    y_pixels = _pixel_axis(grid.y_centers_px, grid.stixel_height_px)
    center_x = float(np.median(np.interp(contour[:, 1], np.arange(x_pixels.size), x_pixels)))
    center_y = float(np.median(np.interp(contour[:, 0], np.arange(y_pixels.size), y_pixels)))
    return RFSuccess(
        center_x_px=center_x,
        center_y_px=center_y,
        spatial_rf=np.asarray(spatial_rf, dtype=np.float32),
        temporal_filter=np.asarray(temporal_filter, dtype=np.float32),
        robust_sd=robust_sd,
        significant_pixel_count=significant_count,
        contour_point_count=int(contour.shape[0]),
        touches_boundary=touches_boundary,
    )


def estimate_condition(stas: np.ndarray, grid: RFGrid) -> ConditionResult:
    cell_count, lag_count, height, width = stas.shape
    centers = np.full((cell_count, 2), np.nan, dtype=np.float64)
    spatial = np.full((cell_count, height, width), np.nan, dtype=np.float32)
    temporal = np.full((cell_count, lag_count), np.nan, dtype=np.float32)
    robust = np.full(cell_count, np.nan, dtype=np.float64)
    significant = np.zeros(cell_count, dtype=np.int64)
    contours = np.zeros(cell_count, dtype=np.int64)
    boundary = np.zeros(cell_count, dtype=bool)
    failures: list[str | None] = []
    for cell_index, sta in enumerate(stas):
        estimate = estimate_rf(sta, grid)
        if isinstance(estimate, RFFailure):
            failures.append(estimate.reason)
            continue
        centers[cell_index] = estimate.center_x_px, estimate.center_y_px
        spatial[cell_index] = estimate.spatial_rf
        temporal[cell_index] = estimate.temporal_filter
        robust[cell_index] = estimate.robust_sd
        significant[cell_index] = estimate.significant_pixel_count
        contours[cell_index] = estimate.contour_point_count
        boundary[cell_index] = estimate.touches_boundary
        failures.append(None)
    return ConditionResult(
        centers, spatial, temporal, robust, significant, contours, boundary, tuple(failures)
    )


__all__ = [
    "RFEstimate",
    "ConditionResult",
    "RFFailure",
    "RFGrid",
    "RFSuccess",
    "batch_spike_triggered_average",
    "estimate_rf",
    "estimate_condition",
    "recreate_binary_white_noise",
    "spike_triggered_average",
]
