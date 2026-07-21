from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from data.cone_response import (
    ConeResponseExport,
    DataContractError,
    load_cone_response,
    validate_response,
)


@dataclass(frozen=True, slots=True)
class ISETBioDatasetConfig:
    h5_path: str | Path
    input_steps: int = 16
    eps: float = 1e-6
    clip: float = 5.0
    allow_fit_stats: bool = False


class ISETBioSample(TypedDict):
    x_cone: torch.Tensor
    target_current: torch.Tensor
    time_index: torch.Tensor


def log_cone_response(response: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if eps <= 0:
        raise DataContractError("eps must be positive")
    return np.log(validate_response(response) + np.float32(eps)).astype(np.float32)


def fit_log_cone_stats(
    paths: Sequence[str | Path],
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise DataContractError("At least one training export is required")

    log_responses: list[np.ndarray] = []
    for export in validate_compatible_cone_exports(paths):
        log_responses.append(log_cone_response(export.response, eps))

    stacked = np.concatenate(log_responses, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    scale = stacked.std(axis=0).astype(np.float32)
    return mean, np.maximum(scale, np.float32(eps))


def validate_compatible_cone_exports(
    paths: Sequence[str | Path],
) -> tuple[ConeResponseExport, ...]:
    if not paths:
        raise DataContractError("At least one cone-response export is required")
    exports = tuple(load_cone_response(Path(path)) for path in paths)
    reference = exports[0]
    reference_dt_ms = float(np.median(np.diff(reference.time_axis_seconds)) * 1000.0)
    for export in exports[1:]:
        if (
            export.positions_degs.shape != reference.positions_degs.shape
            or not np.allclose(
                export.positions_degs,
                reference.positions_degs,
                atol=1e-6,
            )
        ):
            raise DataContractError(
                "Cone-response exports do not use the same cone positions"
            )
        if not np.array_equal(export.cone_types, reference.cone_types):
            raise DataContractError(
                "Cone-response exports use different cone type ordering"
            )
        if not np.isclose(
            export.eccentricity_deg,
            reference.eccentricity_deg,
            rtol=0.0,
            atol=1e-6,
        ):
            raise DataContractError(
                "Cone-response exports use different retinal eccentricities"
            )
        dt_ms = float(np.median(np.diff(export.time_axis_seconds)) * 1000.0)
        if not np.isclose(dt_ms, reference_dt_ms, rtol=1e-6, atol=1e-6):
            raise DataContractError(
                "Cone-response exports use different temporal sampling intervals"
            )
    return exports


def apply_log_cone_stats(
    response: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-6,
    clip: float | None = None,
) -> np.ndarray:
    log_response = log_cone_response(response, eps)
    mean, scale = _validate_stats(mean, scale, log_response.shape[1])
    normalized = (log_response - mean) / scale
    if clip is not None:
        if clip <= 0:
            raise DataContractError("clip must be positive")
        normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32)


def save_log_cone_stats(
    path: str | Path,
    mean: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-6,
) -> None:
    mean, scale = _validate_stats(mean, scale)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, scale=scale, eps=np.float32(eps))


def load_log_cone_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(Path(path)) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        scale = np.asarray(data["scale"], dtype=np.float32)
        eps = float(np.asarray(data["eps"]).reshape(()))
    mean, scale = _validate_stats(mean, scale)
    return mean, scale, eps


class ISETBioDataset(Dataset[ISETBioSample]):
    def __init__(
        self,
        config: ISETBioDatasetConfig,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> None:
        self._config = config
        self._validate_config(config)

        h5_path = Path(config.h5_path)
        export = load_cone_response(h5_path)

        if (mean is None) != (scale is None):
            raise DataContractError("mean and scale must be provided together")
        if mean is None and scale is None:
            if not config.allow_fit_stats:
                raise DataContractError(
                    "Pass train-only normalization stats, or set "
                    "allow_fit_stats=True explicitly for training/smoke tests"
                )
            mean, scale = fit_log_cone_stats([h5_path], config.eps)

        mean, scale = _validate_stats(mean, scale, export.response.shape[1])
        unclipped_contrast = apply_log_cone_stats(
            export.response,
            mean,
            scale,
            config.eps,
            clip=None,
        )
        self._clip_fraction = float(np.mean(np.abs(unclipped_contrast) > config.clip))
        self._contrast = np.clip(
            unclipped_contrast,
            -config.clip,
            config.clip,
        ).astype(np.float32)
        self._normalization_mean = mean
        self._normalization_scale = scale
        self._positions_degs = export.positions_degs
        self._cone_types = export.cone_types
        self._time_axis_seconds = export.time_axis_seconds
        self._eye_trace_degs = export.eye_trace_degs
        self._response_units = export.units
        self._eccentricity_deg = export.eccentricity_deg
        self._length = self._contrast.shape[0] - config.input_steps + 1
        if self._length <= 0:
            raise DataContractError("Sequence is too short for the requested input window")

    @property
    def positions_degs(self) -> np.ndarray:
        return self._positions_degs

    @property
    def cone_types(self) -> np.ndarray:
        return self._cone_types

    @property
    def time_axis_seconds(self) -> np.ndarray:
        return self._time_axis_seconds

    @property
    def dt_ms(self) -> float:
        return float(np.median(np.diff(self._time_axis_seconds)) * 1000.0)

    @property
    def eye_trace_degs(self) -> np.ndarray:
        return self._eye_trace_degs

    @property
    def response_units(self) -> str:
        return self._response_units

    @property
    def eccentricity_deg(self) -> float:
        return self._eccentricity_deg

    @property
    def normalization_mean(self) -> np.ndarray:
        return self._normalization_mean

    @property
    def normalization_scale(self) -> np.ndarray:
        return self._normalization_scale

    @property
    def clip_fraction(self) -> float:
        return self._clip_fraction

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> ISETBioSample:
        if index < 0 or index >= self._length:
            raise IndexError(index)

        anchor = index + self._config.input_steps - 1
        start = anchor - self._config.input_steps + 1
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(self._contrast[anchor]).copy()
        )
        x_cone = torch.from_numpy(
            np.ascontiguousarray(self._contrast[start : anchor + 1]).copy()
        )
        sample: ISETBioSample = {
            "x_cone": x_cone,
            "target_current": target_tensor,
            "time_index": torch.tensor(anchor, dtype=torch.long),
        }
        return sample

    @staticmethod
    def _validate_config(config: ISETBioDatasetConfig) -> None:
        if config.input_steps < 1:
            raise DataContractError("input_steps must be positive")
        if config.eps <= 0:
            raise DataContractError("eps must be positive")
        if config.clip <= 0:
            raise DataContractError("clip must be positive")


def _validate_stats(
    mean: np.ndarray,
    scale: np.ndarray,
    cone_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if mean.ndim != 1 or scale.shape != mean.shape:
        raise DataContractError("Expected per-cone mean and scale with shape [Ncone]")
    if cone_count is not None and mean.shape != (cone_count,):
        raise DataContractError("Normalization stats do not match cone count")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise DataContractError("Normalization stats must be finite with positive scale")
    return mean, scale
