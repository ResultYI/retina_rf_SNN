from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


class DataContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConeResponseExport:
    response: np.ndarray
    positions_degs: np.ndarray
    cone_types: np.ndarray
    time_axis_seconds: np.ndarray
    eye_trace_degs: np.ndarray
    units: str


def load_cone_response(path: str | Path) -> ConeResponseExport:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        version = _decode_text(handle["format_version"])
        if version != "retina-snn-cone-response-v1":
            raise DataContractError(f"Unsupported format_version: {version}")

        shape_values = np.asarray(handle["response_shape_time_cone"][()]).reshape(-1)
        shape = tuple(int(value) for value in shape_values)
        if len(shape) != 2:
            raise DataContractError(f"Expected [T,Ncone] shape metadata, got {shape}")

        response = _logical_array(handle["cone_response"], shape).astype(np.float32)
        positions = _logical_array(handle["cone_positions_degs"], (shape[1], 2)).astype(
            np.float32
        )
        cone_types = np.asarray(handle["cone_types"][()]).reshape(-1).astype(np.uint8)
        time_axis = np.asarray(handle["time_axis_seconds"][()]).reshape(-1).astype(np.float64)
        eye_trace = _logical_array(handle["eye_trace_degs"], (shape[0], 2)).astype(
            np.float32
        )
        units = _decode_text(handle["response_units"])

    if response.shape != shape:
        raise DataContractError(f"Response shape mismatch: {response.shape} versus {shape}")
    if positions.shape != (shape[1], 2):
        raise DataContractError(f"Position shape mismatch: {positions.shape}")
    if cone_types.shape != (shape[1],):
        raise DataContractError(f"Cone type shape mismatch: {cone_types.shape}")
    if time_axis.shape != (shape[0],):
        raise DataContractError(f"Time-axis shape mismatch: {time_axis.shape}")
    if eye_trace.shape != (shape[0], 2):
        raise DataContractError(f"Eye-trace shape mismatch: {eye_trace.shape}")

    return ConeResponseExport(
        response=validate_response(response),
        positions_degs=positions,
        cone_types=cone_types,
        time_axis_seconds=_validate_time_axis(time_axis),
        eye_trace_degs=eye_trace,
        units=units,
    )


def validate_response(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2:
        raise DataContractError("Expected response with shape [T,Ncone]")
    if not np.isfinite(response).all() or np.any(response < 0):
        raise DataContractError("Cone response must be finite and non-negative")
    return response


def _validate_time_axis(time_axis: np.ndarray) -> np.ndarray:
    if not np.isfinite(time_axis).all():
        raise DataContractError("time_axis_seconds must be finite")
    frame_intervals = np.diff(time_axis)
    if np.any(frame_intervals <= 0):
        raise DataContractError("time_axis_seconds must be strictly increasing")
    relative_variation = np.std(frame_intervals) / (np.mean(frame_intervals) + 1e-12)
    if relative_variation > 1e-3:
        raise DataContractError("time_axis_seconds must have a stable frame interval")
    return time_axis


def _decode_text(dataset: h5py.Dataset) -> str:
    value = np.asarray(dataset[()]).astype(np.uint8, copy=False)
    return value.tobytes().decode("utf-8")


def _logical_array(dataset: h5py.Dataset, expected_shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(dataset[()])
    if raw.shape == expected_shape:
        return raw
    if raw.shape == expected_shape[::-1]:
        return raw.T
    raise DataContractError(
        f"{dataset.name}: stored shape {raw.shape} does not match logical "
        f"shape {expected_shape} or its MATLAB/HDF5 reversal"
    )
