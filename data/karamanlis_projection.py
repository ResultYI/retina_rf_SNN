from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StimulusImages:
    values: np.ndarray
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    background: float


@dataclass(frozen=True, slots=True)
class FlashTiming:
    steps: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class ProjectionGeometry:
    image_ids: np.ndarray
    center_pixels: np.ndarray
    pixel_size_um: float
    crop_pixels: int
    pool_factor: int
    retinal_um_per_degree: float


@dataclass(frozen=True, slots=True)
class ConeProjection:
    templates: np.ndarray
    positions_degs: np.ndarray


@dataclass(frozen=True, slots=True)
class KaramanlisProjectionError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def project_achromatic_cone_drive(
    stimulus: StimulusImages,
    timing: FlashTiming,
    geometry: ProjectionGeometry,
) -> ConeProjection:
    grid_size = geometry.crop_pixels // geometry.pool_factor
    half = geometry.crop_pixels // 2
    center_x = int(np.argmin(np.abs(stimulus.x_pixels - geometry.center_pixels[0])))
    center_y = int(np.argmin(np.abs(stimulus.y_pixels - geometry.center_pixels[1])))
    x_slice = slice(center_x - half, center_x + half + 1)
    y_slice = slice(center_y - half, center_y + half + 1)
    if (
        x_slice.start < 0
        or y_slice.start < 0
        or x_slice.stop > stimulus.x_pixels.size
        or y_slice.stop > stimulus.y_pixels.size
    ):
        raise KaramanlisProjectionError(
            "selected cell population lies outside stimulus images"
        )
    patches = stimulus.values[
        geometry.image_ids, x_slice, y_slice
    ].transpose(0, 2, 1).astype(np.float32)
    pooled = patches.reshape(
        geometry.image_ids.size,
        grid_size,
        geometry.pool_factor,
        grid_size,
        geometry.pool_factor,
    ).mean((2, 4))
    contrast = (pooled / 255.0 - stimulus.background) / stimulus.background
    templates = np.zeros(
        (geometry.image_ids.size, timing.steps, grid_size**2), dtype=np.float32
    )
    templates[:, timing.start:timing.stop] = contrast.reshape(
        geometry.image_ids.size, 1, -1
    )
    pooled_x = stimulus.x_pixels[x_slice].reshape(
        grid_size, geometry.pool_factor
    ).mean(1)
    pooled_y = stimulus.y_pixels[y_slice].reshape(
        grid_size, geometry.pool_factor
    ).mean(1)
    yy, xx = np.meshgrid(pooled_y, pooled_x, indexing="ij")
    positions = np.column_stack(
        (
            (xx.reshape(-1) - geometry.center_pixels[0]) * geometry.pixel_size_um,
            (geometry.center_pixels[1] - yy.reshape(-1)) * geometry.pixel_size_um,
        )
    ) / geometry.retinal_um_per_degree
    return ConeProjection(templates, positions)


__all__ = [
    "ConeProjection",
    "FlashTiming",
    "KaramanlisProjectionError",
    "ProjectionGeometry",
    "StimulusImages",
    "project_achromatic_cone_drive",
]
