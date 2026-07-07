from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

REQUIRED_KEYS = (
    "cone_response_lms",
    "cone_response_achromatic",
    "time_axis_seconds",
    "cone_xy_deg",
    "cone_type",
    "eye_movement_xy_deg",
    "config_json",
    "source_image_path",
    "source_image_id",
    "cone_response",
    "cone_positions_degs",
    "cone_types",
    "eye_trace_degs",
    "format_version",
    "response_shape_time_cone",
)
REQUIRED_ATTRS = (
    "dt_ms",
    "field_of_view_deg",
    "eccentricity_deg",
    "mosaic_type",
    "mosaic_seed",
    "stimulus_seed",
    "is_achromatic_stimulus",
    "achromatic_projection_method",
    "ISETBio_git_commit",
    "ISETCam_git_commit",
    "MATLAB_version",
    "generation_date",
)


class H5GateError(RuntimeError):
    def __init__(self, layer: str, message: str) -> None:
        super().__init__(message)
        self.layer = layer


@dataclass(frozen=True, slots=True)
class H5Summary:
    path: Path
    time_steps: int
    cone_count: int
    lms_shape: tuple[int, ...]
    achromatic_shape: tuple[int, ...]
    dt_ms: float


def validate_hdf5(path: Path) -> H5Summary:
    h5py = _import_h5py()
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle]
        missing_attrs = [attr for attr in REQUIRED_ATTRS if attr not in handle.attrs]
        if missing or missing_attrs:
            raise H5GateError(
                "Python readback",
                f"missing keys={missing}, missing attrs={missing_attrs}",
            )
        time_axis = np.asarray(handle["time_axis_seconds"][()]).reshape(-1)
        if time_axis.size < 2 or np.any(np.diff(time_axis) <= 0):
            raise H5GateError("Python readback", "time_axis_seconds invalid")
        dt_ms = float(np.median(np.diff(time_axis)) * 1000.0)
        attr_dt = float(np.asarray(handle.attrs["dt_ms"]).reshape(-1)[0])
        if abs(dt_ms - attr_dt) > 1e-6:
            raise H5GateError("Python readback", "dt_ms attribute mismatch")
        achromatic = np.asarray(handle["cone_response_achromatic"][()])
        lms = np.asarray(handle["cone_response_lms"][()])
        cone_count = _validate_response_shapes(achromatic, lms, time_axis.size)
    _validate_current_dataset_aliases(path)
    return H5Summary(path, time_axis.size, cone_count, lms.shape, achromatic.shape, dt_ms)


def compare_reproducible(first: Path, second: Path) -> None:
    h5py = _import_h5py()
    keys = (
        "cone_response_lms",
        "cone_response_achromatic",
        "time_axis_seconds",
        "cone_xy_deg",
        "cone_type",
        "eye_movement_xy_deg",
    )
    with h5py.File(first, "r") as left, h5py.File(second, "r") as right:
        for key in keys:
            if not np.allclose(np.asarray(left[key][()]), np.asarray(right[key][()])):
                raise H5GateError("Python readback", f"not reproducible: {key}")


def _import_h5py() -> ModuleType:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise H5GateError("Python readback", f"h5py unavailable: {exc}") from exc
    return h5py


def _validate_response_shapes(
    achromatic: np.ndarray, lms: np.ndarray, time_count: int
) -> int:
    if not np.isfinite(achromatic).all() or not np.isfinite(lms).all():
        raise H5GateError("Python readback", "cone_response contains NaN/Inf")
    if not np.any(achromatic):
        raise H5GateError("Python readback", "cone_response is all zero")
    if achromatic.ndim != 2 or lms.ndim != 3:
        raise H5GateError("Python readback", "unexpected response rank")
    if achromatic.shape[0] == time_count:
        cone_count = achromatic.shape[1]
    elif achromatic.shape[1] == time_count:
        cone_count = achromatic.shape[0]
    else:
        raise H5GateError("Python readback", "achromatic response/time mismatch")
    if time_count not in lms.shape or 3 not in lms.shape:
        raise H5GateError("Python readback", "LMS response/time mismatch")
    return int(cone_count)


def _validate_current_dataset_aliases(path: Path) -> None:
    try:
        from data.cone_response import load_cone_response

        load_cone_response(path)
    except (ImportError, ValueError, OSError, KeyError) as exc:
        raise H5GateError("Python Dataset readback", str(exc)) from exc
