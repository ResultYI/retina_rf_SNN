from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

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
    eccentricity_deg: float
    source_id: str | None = None
    source_movie_id: str | None = None
    stimulus_source_kind: str | None = None
    metadata_config: str | None = None


def load_cone_response(path: str | Path) -> ConeResponseExport:
    path = Path(path)
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        version = _decode_text(handle["format_version"])
        if version != "retina-snn-cone-response-v1":
            raise DataContractError(f"Unsupported format_version: {version}")

        shape_values = np.asarray(handle["response_shape_time_cone"][()]).reshape(-1)
        shape = tuple(int(value) for value in shape_values)
        if len(shape) != 2:
            raise DataContractError(f"Expected [T,Ncone] shape metadata, got {shape}")

        response = _logical_array(
            _first_dataset(handle, ("cone_response_achromatic", "cone_response")),
            shape,
        ).astype(np.float32)
        positions = _logical_array(
            _first_dataset(handle, ("cone_xy_deg", "cone_positions_degs")),
            (shape[1], 2),
        ).astype(
            np.float32
        )
        cone_types = np.asarray(
            _first_dataset(handle, ("cone_type", "cone_types"))[()]
        ).reshape(-1).astype(np.uint8)
        time_axis = (
            np.asarray(handle["time_axis_seconds"][()])
            .reshape(-1)
            .astype(np.float64)
        )
        eye_trace = _logical_array(
            _first_dataset(handle, ("eye_movement_xy_deg", "eye_trace_degs")),
            (shape[0], 2),
        ).astype(
            np.float32
        )
        units = _decode_text(handle["response_units"])
        if "eccentricity_deg" not in handle.attrs:
            raise DataContractError("Missing required eccentricity_deg attribute")
        eccentricity_values = np.asarray(
            handle.attrs["eccentricity_deg"], dtype=np.float64
        ).reshape(-1)
        if eccentricity_values.size not in {1, 2} or not np.isfinite(
            eccentricity_values
        ).all():
            raise DataContractError(
                "eccentricity_deg must be a finite scalar or [x_deg,y_deg]"
            )
        if eccentricity_values.size == 1:
            eccentricity_deg = float(eccentricity_values[0])
            if eccentricity_deg < 0:
                raise DataContractError("scalar eccentricity_deg must be non-negative")
        else:
            eccentricity_deg = float(np.linalg.norm(eccentricity_values))
        metadata_config = _optional_text(handle, ("metadata/config", "config_json"))
        source_movie_id = _optional_text(handle, ("source_movie_id",))
        source_id = source_movie_id or _optional_text(
            handle,
            ("source_image_id", "source_id"),
        )
        stimulus_source_kind = _optional_attribute_text(
            handle,
            "stimulus_source_kind",
        )

    if response.shape != shape:
        raise DataContractError(
            f"Response shape mismatch: {response.shape} versus {shape}"
        )
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
        eccentricity_deg=eccentricity_deg,
        source_id=source_id,
        source_movie_id=source_movie_id,
        stimulus_source_kind=stimulus_source_kind,
        metadata_config=metadata_config,
    )


def validate_formal_stimulus_splits(
    train: Sequence[ConeResponseExport],
    validation: Sequence[ConeResponseExport],
    test: Sequence[ConeResponseExport] = (),
) -> None:
    split_ids = (
        _formal_source_ids("train", train),
        _formal_source_ids("validation", validation),
        _formal_source_ids("test", test),
    )
    if (
        split_ids[0] & split_ids[1]
        or split_ids[0] & split_ids[2]
        or split_ids[1] & split_ids[2]
    ):
        raise DataContractError("Formal stimulus splits must be source-disjoint")


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


def _optional_text(handle: h5py.File, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in handle:
            return _decode_text(handle[name])
    return None


def _optional_attribute_text(handle: h5py.File, name: str) -> str | None:
    if name not in handle.attrs:
        return None
    value = np.asarray(handle.attrs[name]).astype(str).reshape(-1)
    return None if value.size == 0 else str(value[0])


def _formal_source_ids(
    split_name: str,
    exports: Sequence[ConeResponseExport],
) -> set[str]:
    source_ids: set[str] = set()
    for export in exports:
        source_kind = export.stimulus_source_kind
        if source_kind == "natural_video":
            source_id = export.source_movie_id
        elif source_kind in {
            "natural_image_microdrift",
            "natural_image_fixational_eye_movement",
        }:
            source_id = export.source_id
            if np.all(np.ptp(export.eye_trace_degs, axis=0) <= 0):
                raise DataContractError(
                    f"{split_name} natural-image export needs a non-static eye trace"
                )
        else:
            raise DataContractError(
                f"{split_name} export must declare a supported natural stimulus kind"
            )
        if not source_id:
            raise DataContractError(
                f"{split_name} formal export needs a stable source id"
            )
        source_ids.add(source_id)
    return source_ids


def _first_dataset(
    handle: h5py.File,
    names: tuple[str, ...],
) -> h5py.Dataset:
    for name in names:
        if name in handle:
            return handle[name]
    raise DataContractError(f"Missing required dataset; expected one of {names}")


def _logical_array(
    dataset: h5py.Dataset,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    raw = np.asarray(dataset[()])
    if raw.shape == expected_shape:
        return raw
    if raw.shape == expected_shape[::-1]:
        return raw.T
    raise DataContractError(
        f"{dataset.name}: stored shape {raw.shape} does not match logical "
        f"shape {expected_shape} or its MATLAB/HDF5 reversal"
    )


def _import_h5py() -> ModuleType:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise DataContractError(f"h5py unavailable: {exc}") from exc
    return h5py
