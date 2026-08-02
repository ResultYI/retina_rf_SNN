from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from data.rgc_response import RGCResponseContractError


@dataclass(frozen=True, slots=True)
class TeacherInputNormalization:
    input_mean: np.ndarray
    input_std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.input_mean, dtype=np.float32).reshape(-1).copy()
        std = np.asarray(self.input_std, dtype=np.float32).reshape(-1).copy()
        if mean.shape != std.shape or mean.size == 0:
            raise RGCResponseContractError("teacher normalization shape is invalid")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RGCResponseContractError("teacher normalization must be finite")
        if np.any(std <= 0):
            raise RGCResponseContractError("teacher normalization std must be positive")
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "input_mean", mean)
        object.__setattr__(self, "input_std", std)

    def normalize(self, cones: np.ndarray) -> np.ndarray:
        return (cones - self.input_mean) / self.input_std

    def matches(self, other: TeacherInputNormalization) -> bool:
        return bool(
            np.array_equal(self.input_mean, other.input_mean)
            and np.array_equal(self.input_std, other.input_std)
        )


@dataclass(frozen=True, slots=True)
class TeacherRFMetadata:
    static_kernel: np.ndarray
    context_kernel_low: np.ndarray
    context_kernel_high: np.ndarray
    context_gain_envelope: np.ndarray | None
    cell_group_ids: tuple[str, ...] = ()
    cell_replicate_ids: tuple[str, ...] = ()
    component_ids: tuple[str, ...] = ()
    revision: str | None = None
    generation_seed: int | None = None
    residual_seed: int | None = None
    cells_per_type_polarity: int | None = None
    residual_bound: float | None = None
    context_high_scale: np.ndarray | None = None
    context_gain_population_component: np.ndarray | None = None
    context_gain_type_component: np.ndarray | None = None
    context_gain_polarity_component: np.ndarray | None = None
    context_gain_cell_residual: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("static_kernel", self.static_kernel),
            ("context_kernel_low", self.context_kernel_low),
            ("context_kernel_high", self.context_kernel_high),
        ):
            if value.ndim != 3 or not np.isfinite(value).all():
                raise RGCResponseContractError(f"{name} teacher kernel is invalid")
        cell_count = self.static_kernel.shape[0]
        for name, values in (
            ("cell_group_ids", self.cell_group_ids),
            ("cell_replicate_ids", self.cell_replicate_ids),
        ):
            if values and len(values) != cell_count:
                raise RGCResponseContractError(f"{name} teacher metadata is invalid")
        for name, value in (
            ("context_high_scale", self.context_high_scale),
            (
                "context_gain_population_component",
                self.context_gain_population_component,
            ),
            ("context_gain_type_component", self.context_gain_type_component),
            ("context_gain_polarity_component", self.context_gain_polarity_component),
            ("context_gain_cell_residual", self.context_gain_cell_residual),
        ):
            if value is not None and (
                value.shape != (cell_count,) or not np.isfinite(value).all()
            ):
                raise RGCResponseContractError(f"{name} teacher metadata is invalid")


def fit_teacher_input_normalization(
    cone_sequences: np.ndarray,
) -> TeacherInputNormalization:
    if cone_sequences.ndim != 3:
        raise RGCResponseContractError(
            "teacher normalization fit requires [stimulus,time,cone]"
        )
    mean = cone_sequences.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = cone_sequences.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    return TeacherInputNormalization(mean, np.maximum(std, 1e-6))


def load_teacher_input_normalization(
    path: str | Path,
    cone_count: int,
) -> TeacherInputNormalization | None:
    with h5py.File(Path(path), "r") as handle:
        if "teacher" not in handle:
            return None
        has_mean = "teacher/input_mean" in handle
        has_std = "teacher/input_std" in handle
        if not has_mean or not has_std:
            raise RGCResponseContractError(
                "teacher normalization must include input_mean and input_std"
            )
        normalization = TeacherInputNormalization(
            np.asarray(handle["teacher/input_mean"][()]),
            np.asarray(handle["teacher/input_std"][()]),
        )
    if normalization.input_mean.shape != (cone_count,):
        raise RGCResponseContractError(
            "teacher normalization shape does not match cone count"
        )
    return normalization


def load_teacher_rf_metadata(path: str | Path) -> TeacherRFMetadata | None:
    with h5py.File(Path(path), "r") as handle:
        keys = (
            "teacher/static_kernel",
            "teacher/context_kernel_low",
            "teacher/context_kernel_high",
        )
        present = tuple(key in handle for key in keys)
        if not any(present):
            return None
        if not all(present):
            raise RGCResponseContractError("synthetic teacher RF metadata is partial")
        envelope = (
            np.asarray(handle["teacher/context_gain_envelope"][()], dtype=np.float32)
            if "teacher/context_gain_envelope" in handle
            else None
        )
        return TeacherRFMetadata(
            static_kernel=np.asarray(handle[keys[0]][()], dtype=np.float32),
            context_kernel_low=np.asarray(handle[keys[1]][()], dtype=np.float32),
            context_kernel_high=np.asarray(handle[keys[2]][()], dtype=np.float32),
            context_gain_envelope=envelope,
            cell_group_ids=_optional_text_vector(handle, "teacher/cell_group_id"),
            cell_replicate_ids=_optional_text_vector(
                handle,
                "teacher/cell_replicate_id",
            ),
            component_ids=_optional_text_vector(handle, "teacher/component_id"),
            revision=_optional_scalar_text(handle, "teacher/revision"),
            generation_seed=_optional_int_scalar(handle, "teacher/generation_seed"),
            residual_seed=_optional_int_scalar(handle, "teacher/residual_seed"),
            cells_per_type_polarity=_optional_int_scalar(
                handle,
                "teacher/cells_per_type_polarity",
            ),
            residual_bound=_optional_float_scalar(handle, "teacher/residual_bound"),
            context_high_scale=_optional_float_vector(
                handle,
                "teacher/context_high_scale",
            ),
            context_gain_population_component=_optional_float_vector(
                handle,
                "teacher/context_gain_population_component",
            ),
            context_gain_type_component=_optional_float_vector(
                handle,
                "teacher/context_gain_type_component",
            ),
            context_gain_polarity_component=_optional_float_vector(
                handle,
                "teacher/context_gain_polarity_component",
            ),
            context_gain_cell_residual=_optional_float_vector(
                handle,
                "teacher/context_gain_cell_residual",
            ),
        )


def _optional_int_scalar(handle: h5py.File, key: str) -> int | None:
    return int(np.asarray(handle[key][()]).reshape(-1)[0]) if key in handle else None


def _optional_float_scalar(handle: h5py.File, key: str) -> float | None:
    return float(np.asarray(handle[key][()]).reshape(-1)[0]) if key in handle else None


def _optional_float_vector(handle: h5py.File, key: str) -> np.ndarray | None:
    return (
        np.asarray(handle[key][()], dtype=np.float32).reshape(-1)
        if key in handle
        else None
    )


def _optional_scalar_text(handle: h5py.File, key: str) -> str | None:
    values = _optional_text_vector(handle, key)
    return None if not values else values[0]


def _optional_text_vector(handle: h5py.File, key: str) -> tuple[str, ...]:
    if key not in handle:
        return ()
    values = np.asarray(handle[key][()]).reshape(-1)
    return tuple(_decode_text(value) for value in values)


def _decode_text(value: str | bytes | np.ndarray | np.generic) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.reshape(-1).tobytes().decode("utf-8")
    return str(array.item())


__all__ = [
    "TeacherInputNormalization",
    "TeacherRFMetadata",
    "fit_teacher_input_normalization",
    "load_teacher_input_normalization",
    "load_teacher_rf_metadata",
]
