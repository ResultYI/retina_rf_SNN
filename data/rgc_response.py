from __future__ import annotations
# noqa: SIZE_OK — one HDF5 response boundary keeps parsing and validation atomic.

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np

from data.input_identity import (
    DatasetKind,
    InputIdentity,
    InputIdentityError,
    legacy_input_identity,
)


TextValue: TypeAlias = str | bytes | np.ndarray | np.generic | None


class RGCResponseContractError(ValueError):
    pass


@unique
class ResponseTargetKind(StrEnum):
    BERNOULLI = "bernoulli"
    POISSON = "poisson"


@dataclass(frozen=True, slots=True)
class CellMetadata:
    ids: tuple[str, ...]
    type_ids: tuple[str, ...]
    polarities: np.ndarray
    positions_degs: np.ndarray
    eccentricities_deg: np.ndarray


@dataclass(frozen=True, slots=True)
class RGCResponseSession:
    cone_response: np.ndarray
    spike_counts: np.ndarray
    valid_mask: np.ndarray
    time_axis_seconds: np.ndarray
    cone_positions_degs: np.ndarray
    cells: CellMetadata
    source_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    target_kind: ResponseTargetKind
    path: Path
    input_identity: InputIdentity = field(default_factory=legacy_input_identity)

    @property
    def dt_ms(self) -> float:
        return float(np.median(np.diff(self.time_axis_seconds)) * 1000.0)


def load_rgc_response(path: str | Path) -> RGCResponseSession:
    source_path = Path(path)
    with h5py.File(source_path, "r") as handle:
        version = _read_scalar_text(handle, "format_version")
        if version not in {"retina-rgc-response-v1", "retina-rgc-response-v2"}:
            raise RGCResponseContractError(f"Unsupported format_version: {version}")
        kind = _target_kind(handle.attrs.get("response_target_kind"))
        cone = np.asarray(_required(handle, "cone_response"), dtype=np.float32)
        spikes = np.asarray(_required(handle, "spike_counts"), dtype=np.float32)
        mask = np.asarray(_required(handle, "valid_mask"), dtype=bool)
        time_axis = np.asarray(
            _required(handle, "time_axis_seconds"), dtype=np.float64
        ).reshape(-1)
        cone_positions = np.asarray(
            _required(handle, "cone/position_degs"), dtype=np.float32
        )
        cell_ids = _read_text_vector(handle, "cell/id")
        type_ids = _read_text_vector(handle, "cell/type_id")
        polarities = np.asarray(
            _required(handle, "cell/polarity"), dtype=np.int64
        ).reshape(-1)
        cell_positions = np.asarray(
            _required(handle, "cell/position_degs"), dtype=np.float32
        )
        eccentricities = np.asarray(
            _required(handle, "cell/eccentricity_deg"), dtype=np.float32
        ).reshape(-1)
        source_ids = _read_text_vector(handle, "stimulus/source_id")
        context_ids = _read_text_vector(handle, "stimulus/context_id")
        input_identity = (
            _read_input_identity(handle)
            if version == "retina-rgc-response-v2"
            else legacy_input_identity()
        )

    _validate_shapes(
        cone,
        spikes,
        mask,
        time_axis,
        cone_positions,
        cell_ids,
        type_ids,
        polarities,
        cell_positions,
        eccentricities,
        source_ids,
        context_ids,
    )
    _validate_values(
        cone,
        spikes,
        mask,
        time_axis,
        cone_positions,
        cell_positions,
        eccentricities,
        polarities,
        kind,
    )
    _validate_input_identity(
        input_identity,
        stimulus_count=cone.shape[0],
        cone_count=cone.shape[2],
    )
    return RGCResponseSession(
        cone_response=cone,
        spike_counts=spikes,
        valid_mask=mask,
        time_axis_seconds=time_axis,
        cone_positions_degs=cone_positions,
        cells=CellMetadata(
            ids=cell_ids,
            type_ids=type_ids,
            polarities=polarities,
            positions_degs=cell_positions,
            eccentricities_deg=eccentricities,
        ),
        source_ids=source_ids,
        context_ids=context_ids,
        target_kind=kind,
        path=source_path,
        input_identity=input_identity,
    )


def validate_response_splits(
    train: Sequence[RGCResponseSession],
    validation: Sequence[RGCResponseSession],
    test: Sequence[RGCResponseSession] = (),
    *,
    sequence_steps: int | None = None,
) -> None:
    if not train or not validation:
        raise RGCResponseContractError("train and validation splits must be non-empty")
    groups = tuple(_source_ids(split) for split in (train, validation, test))
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RGCResponseContractError("Response splits must be source-disjoint")
    reference = train[0]
    for session in (*train, *validation, *test):
        if session.target_kind is not reference.target_kind:
            raise RGCResponseContractError("All splits must use one target kind")
        if (
            session.input_identity.compatibility_key()
            != reference.input_identity.compatibility_key()
        ):
            raise RGCResponseContractError(
                "All splits must use one compatible input identity"
            )
        if session.cells.ids != reference.cells.ids:
            raise RGCResponseContractError("Cell order must match across response files")
        if session.cells.type_ids != reference.cells.type_ids:
            raise RGCResponseContractError("Cell types must match across response files")
        if not np.array_equal(session.cells.polarities, reference.cells.polarities):
            raise RGCResponseContractError("Cell polarities must match across response files")
        if not _same_values(
            session.cells.positions_degs,
            reference.cells.positions_degs,
        ):
            raise RGCResponseContractError("Cell positions must match across response files")
        if not _same_values(
            session.cells.eccentricities_deg,
            reference.cells.eccentricities_deg,
        ):
            raise RGCResponseContractError(
                "Cell eccentricities must match across response files"
            )
        if not _same_values(
            session.cone_positions_degs,
            reference.cone_positions_degs,
        ):
            raise RGCResponseContractError("Cone geometry must match across response files")
        if not _same_values(
            session.time_axis_seconds[:sequence_steps],
            reference.time_axis_seconds[:sequence_steps],
        ):
            raise RGCResponseContractError("Time axes must match across response files")
    fingerprint_groups = tuple(
        _source_fingerprints(split) for split in (train, validation, test)
    )
    if (
        fingerprint_groups[0] & fingerprint_groups[1]
        or fingerprint_groups[0] & fingerprint_groups[2]
        or fingerprint_groups[1] & fingerprint_groups[2]
    ):
        raise RGCResponseContractError(
            "Response splits must be source-content-disjoint"
        )


def _same_values(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.allclose(left, right))


def _validate_shapes(
    cone: np.ndarray,
    spikes: np.ndarray,
    mask: np.ndarray,
    time_axis: np.ndarray,
    cone_positions: np.ndarray,
    cell_ids: tuple[str, ...],
    type_ids: tuple[str, ...],
    polarities: np.ndarray,
    cell_positions: np.ndarray,
    eccentricities: np.ndarray,
    source_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> None:
    if cone.ndim != 3:
        raise RGCResponseContractError("cone_response must have shape [stimulus,time,cone]")
    if spikes.ndim != 4:
        raise RGCResponseContractError(
            "spike_counts must have shape [stimulus,trial,time,cell]"
        )
    stimulus_count, time_count, cone_count = cone.shape
    if spikes.shape[0] != stimulus_count or spikes.shape[2] != time_count:
        raise RGCResponseContractError("Cone and spike stimulus/time shapes disagree")
    cell_count = spikes.shape[3]
    expected = (cell_count,)
    if mask.shape != spikes.shape:
        raise RGCResponseContractError("valid_mask must match spike_counts")
    if time_axis.shape != (time_count,):
        raise RGCResponseContractError("time_axis_seconds length is invalid")
    if cone_positions.shape != (cone_count, 2):
        raise RGCResponseContractError("cone positions must have shape [cone,2]")
    if not all(len(values) == cell_count for values in (cell_ids, type_ids)):
        raise RGCResponseContractError("Cell metadata length is invalid")
    if polarities.shape != expected or eccentricities.shape != expected:
        raise RGCResponseContractError("Cell vector metadata length is invalid")
    if cell_positions.shape != (cell_count, 2):
        raise RGCResponseContractError("cell positions must have shape [cell,2]")
    if len(source_ids) != stimulus_count or len(context_ids) != stimulus_count:
        raise RGCResponseContractError("Stimulus metadata length is invalid")
    if len(set(cell_ids)) != cell_count:
        raise RGCResponseContractError("Cell identifiers must be unique")


def _validate_values(
    cone: np.ndarray,
    spikes: np.ndarray,
    mask: np.ndarray,
    time_axis: np.ndarray,
    cone_positions: np.ndarray,
    cell_positions: np.ndarray,
    eccentricities: np.ndarray,
    polarities: np.ndarray,
    kind: ResponseTargetKind,
) -> None:
    if (
        not np.isfinite(cone).all()
        or not np.isfinite(spikes).all()
        or not np.isfinite(cone_positions).all()
        or not np.isfinite(cell_positions).all()
        or not np.isfinite(eccentricities).all()
    ):
        raise RGCResponseContractError("Responses must be finite")
    if np.any(spikes < 0):
        raise RGCResponseContractError("spike_counts must be non-negative")
    if kind is ResponseTargetKind.BERNOULLI and not np.all(
        np.logical_or(spikes == 0, spikes == 1)
    ):
        raise RGCResponseContractError("Bernoulli spike_counts must be binary")
    if kind is ResponseTargetKind.POISSON and not np.all(spikes == np.floor(spikes)):
        raise RGCResponseContractError("Poisson spike_counts must be integer valued")
    if not np.all(np.isin(polarities, (0, 1))):
        raise RGCResponseContractError("cell polarity must contain only 0=ON or 1=OFF")
    if time_axis.size < 2 or not np.all(np.diff(time_axis) > 0):
        raise RGCResponseContractError("time_axis_seconds must be strictly increasing")
    intervals = np.diff(time_axis)
    if float(intervals.std() / (np.median(intervals) + 1e-12)) > 1e-3:
        raise RGCResponseContractError(
            "time_axis_seconds must have a stable frame interval"
        )
    if not mask.any(axis=(0, 1, 2)).all():
        raise RGCResponseContractError("Every cell needs at least one valid target")
    invalid_seen = np.maximum.accumulate(~mask, axis=2)
    if np.any(invalid_seen & mask):
        raise RGCResponseContractError(
            "valid_mask cannot become true after missing spike history"
        )


def _validate_input_identity(
    identity: InputIdentity,
    *,
    stimulus_count: int,
    cone_count: int,
) -> None:
    if identity.dataset_kind is DatasetKind.LEGACY_UNSPECIFIED:
        return
    if len(identity.stimulus_source_fingerprints) != stimulus_count:
        raise RGCResponseContractError(
            "Input identity source fingerprints must match stimulus count"
        )
    if len(identity.cone_types) != cone_count:
        raise RGCResponseContractError(
            "Input identity cone types must match cone count"
        )


def _read_input_identity(handle: h5py.File) -> InputIdentity:
    try:
        return InputIdentity(
            dataset_kind=DatasetKind(_read_scalar_text(handle, "input/dataset_kind")),
            species=_read_scalar_text(handle, "input/species"),
            optics_species=_read_scalar_text(handle, "input/optics_species"),
            mosaic_species=_read_scalar_text(handle, "input/mosaic_species"),
            photoreceptor_mode=_read_scalar_text(handle, "input/photoreceptor_mode"),
            chromatic_mode=_read_scalar_text(handle, "input/chromatic_mode"),
            light_level=_read_scalar_text(handle, "input/light_level"),
            mean_luminance_cd_m2=float(
                np.asarray(_required(handle, "input/mean_luminance_cd_m2")).item()
            ),
            cone_types=tuple(
                int(value)
                for value in np.asarray(
                    _required(handle, "input/cone_type")
                ).reshape(-1)
            ),
            response_units=_read_scalar_text(handle, "input/response_units"),
            mosaic_id=_read_scalar_text(handle, "input/cone_mosaic_id"),
            mosaic_fingerprint=_read_scalar_text(
                handle,
                "input/cone_mosaic_fingerprint",
            ),
            stimulus_source_fingerprints=_read_text_vector(
                handle,
                "stimulus/source_content_sha256",
            ),
            generator_name=_read_scalar_text(handle, "input/generator_name"),
            generator_revision=_read_scalar_text(handle, "input/generator_revision"),
            cone_bin_reference=_read_scalar_text(
                handle,
                "input/cone_bin_reference",
            ),
            spike_bin_reference=_read_scalar_text(
                handle,
                "input/spike_bin_reference",
            ),
            stimulus_to_spike_offset_bins=int(
                np.asarray(
                    _required(handle, "input/stimulus_to_spike_offset_bins")
                ).item()
            ),
        )
    except (InputIdentityError, ValueError) as exc:
        raise RGCResponseContractError(f"Invalid input identity: {exc}") from exc


def _required(handle: h5py.File, key: str) -> h5py.Dataset:
    if key not in handle:
        raise RGCResponseContractError(f"Missing required dataset: {key}")
    return handle[key]


def _read_scalar_text(handle: h5py.File, key: str) -> str:
    return _decode_text(_required(handle, key)[()])


def _read_text_vector(handle: h5py.File, key: str) -> tuple[str, ...]:
    values = np.asarray(_required(handle, key)[()]).reshape(-1)
    return tuple(_decode_text(value) for value in values)


def _decode_text(value: TextValue) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.reshape(-1).tobytes().decode("utf-8")
    return str(array.item())


def _target_kind(value: TextValue) -> ResponseTargetKind:
    try:
        return ResponseTargetKind(_decode_text(value))
    except (TypeError, ValueError) as exc:
        raise RGCResponseContractError(
            "response_target_kind must be bernoulli or poisson"
        ) from exc


def _source_ids(sessions: Sequence[RGCResponseSession]) -> set[str]:
    return {source_id for session in sessions for source_id in session.source_ids}


def _source_fingerprints(sessions: Sequence[RGCResponseSession]) -> set[str]:
    return {
        fingerprint
        for session in sessions
        for fingerprint in session.input_identity.stimulus_source_fingerprints
    }


__all__ = [
    "CellMetadata",
    "InputIdentity",
    "RGCResponseContractError",
    "RGCResponseSession",
    "ResponseTargetKind",
    "load_rgc_response",
    "validate_response_splits",
]
