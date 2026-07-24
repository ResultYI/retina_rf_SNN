from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from data.rgc_response import RGCResponseSession
from data.synthetic_teacher import TeacherInputNormalization


def write_rgc_response(
    path: str | Path,
    session: RGCResponseSession,
    *,
    teacher_kernels: dict[str, np.ndarray] | None = None,
    teacher_normalization: TeacherInputNormalization | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(destination, "w") as handle:
        handle.create_dataset(
            "format_version",
            data=np.frombuffer(b"retina-rgc-response-v1", dtype=np.uint8),
        )
        handle.attrs["response_target_kind"] = session.target_kind.value
        handle.create_dataset("cone_response", data=session.cone_response)
        handle.create_dataset("spike_counts", data=session.spike_counts)
        handle.create_dataset(
            "valid_mask", data=session.valid_mask.astype(np.uint8)
        )
        handle.create_dataset("time_axis_seconds", data=session.time_axis_seconds)
        handle.create_dataset(
            "cone/position_degs", data=session.cone_positions_degs
        )
        handle.create_dataset(
            "cell/id", data=np.asarray(session.cells.ids, dtype=string_dtype)
        )
        handle.create_dataset(
            "cell/type_id",
            data=np.asarray(session.cells.type_ids, dtype=string_dtype),
        )
        handle.create_dataset("cell/polarity", data=session.cells.polarities)
        handle.create_dataset(
            "cell/position_degs", data=session.cells.positions_degs
        )
        handle.create_dataset(
            "cell/eccentricity_deg", data=session.cells.eccentricities_deg
        )
        handle.create_dataset(
            "stimulus/source_id",
            data=np.asarray(session.source_ids, dtype=string_dtype),
        )
        handle.create_dataset(
            "stimulus/context_id",
            data=np.asarray(session.context_ids, dtype=string_dtype),
        )
        if teacher_kernels or teacher_normalization is not None:
            group = handle.create_group("teacher")
            if teacher_normalization is not None:
                group.create_dataset(
                    "input_mean", data=teacher_normalization.input_mean
                )
                group.create_dataset(
                    "input_std", data=teacher_normalization.input_std
                )
        if teacher_kernels:
            for name, values in teacher_kernels.items():
                group.create_dataset(name, data=values)


__all__ = ["write_rgc_response"]
