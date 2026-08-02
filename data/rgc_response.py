from __future__ import annotations
# noqa: SIZE_OK — one HDF5 response boundary keeps parsing and validation atomic.

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique
from pathlib import Path
from typing import Final, TypeAlias, assert_never

import h5py
import numpy as np

from data.input_identity import (
    DatasetKind,
    InputIdentity,
    InputIdentityError,
    legacy_input_identity,
)


TextValue: TypeAlias = str | bytes | np.ndarray | np.generic | None
FingerprintFieldValue: TypeAlias = str | tuple[str, ...] | np.ndarray
_FRAME_SIZE_BYTES: Final = 8


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
class TeacherIdentityMetadata:
    revision: str
    generation_seed: int
    residual_seed: int
    cells_per_type_polarity: int
    residual_bound: float
    cell_group_ids: tuple[str, ...]
    cell_replicate_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    context_high_scale: np.ndarray
    context_gain_cell_residual: np.ndarray

    def matches(self, other: TeacherIdentityMetadata) -> bool:
        return bool(
            self.revision == other.revision
            and self.generation_seed == other.generation_seed
            and self.residual_seed == other.residual_seed
            and self.cells_per_type_polarity == other.cells_per_type_polarity
            and np.isclose(self.residual_bound, other.residual_bound)
            and self.cell_group_ids == other.cell_group_ids
            and self.cell_replicate_ids == other.cell_replicate_ids
            and self.component_ids == other.component_ids
            and np.array_equal(self.context_high_scale, other.context_high_scale)
            and np.array_equal(
                self.context_gain_cell_residual,
                other.context_gain_cell_residual,
            )
        )

    def identity_bytes(self) -> bytes:
        fields: tuple[tuple[str, FingerprintFieldValue], ...] = (
            ("revision", self.revision),
            ("generation_seed", str(self.generation_seed)),
            ("residual_seed", str(self.residual_seed)),
            ("cells_per_type_polarity", str(self.cells_per_type_polarity)),
            ("residual_bound", repr(self.residual_bound)),
            ("cell_group_ids", self.cell_group_ids),
            ("cell_replicate_ids", self.cell_replicate_ids),
            ("component_ids", self.component_ids),
            ("context_high_scale", self.context_high_scale),
            ("context_gain_cell_residual", self.context_gain_cell_residual),
        )
        return b"".join(fingerprint_field_bytes(tag, value) for tag, value in fields)


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
    teacher_identity: TeacherIdentityMetadata | None = None

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
        teacher_identity = _read_teacher_identity(handle)

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
    _validate_teacher_identity(teacher_identity, cell_count=spikes.shape[3])
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
        teacher_identity=teacher_identity,
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
        if not _same_teacher_identity(session.teacher_identity, reference.teacher_identity):
            raise RGCResponseContractError(
                "Synthetic teacher identity must match across response files"
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


def _same_teacher_identity(
    left: TeacherIdentityMetadata | None,
    right: TeacherIdentityMetadata | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return left.matches(right)


def fingerprint_field_bytes(tag: str, value: FingerprintFieldValue) -> bytes:
    payload = bytearray(b"rgc-fingerprint-field-v1")
    _append_framed_bytes(payload, tag.encode("utf-8"))
    match value:
        case str():
            _append_framed_bytes(payload, b"utf8")
            _append_shape(payload, ())
            value_bytes = value.encode("utf-8")
        case tuple():
            _append_framed_bytes(payload, b"utf8")
            _append_shape(payload, (len(value),))
            values = bytearray()
            for item in value:
                _append_framed_bytes(values, item.encode("utf-8"))
            value_bytes = bytes(values)
        case np.ndarray():
            array = np.ascontiguousarray(value)
            _append_framed_bytes(payload, array.dtype.str.encode("ascii"))
            _append_shape(payload, array.shape)
            value_bytes = array.tobytes()
        case unreachable:
            assert_never(unreachable)
    _append_framed_bytes(payload, value_bytes)
    return bytes(payload)


def _append_framed_bytes(payload: bytearray, value: bytes) -> None:
    payload.extend(len(value).to_bytes(_FRAME_SIZE_BYTES, "big", signed=False))
    payload.extend(value)


def _append_shape(payload: bytearray, shape: tuple[int, ...]) -> None:
    payload.extend(len(shape).to_bytes(_FRAME_SIZE_BYTES, "big", signed=False))
    for value in shape:
        payload.extend(int(value).to_bytes(_FRAME_SIZE_BYTES, "big", signed=False))


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


def _validate_teacher_identity(
    identity: TeacherIdentityMetadata | None,
    *,
    cell_count: int,
) -> None:
    if identity is None:
        return
    if (
        len(identity.cell_group_ids) != cell_count
        or len(identity.cell_replicate_ids) != cell_count
        or identity.context_high_scale.shape != (cell_count,)
        or identity.context_gain_cell_residual.shape != (cell_count,)
    ):
        raise RGCResponseContractError(
            "Synthetic teacher identity vectors must match cell count"
        )
    text_values = (
        identity.revision,
        *identity.cell_group_ids,
        *identity.cell_replicate_ids,
        *identity.component_ids,
    )
    if any(not value.strip() for value in text_values):
        raise RGCResponseContractError("Synthetic teacher identity text is invalid")
    if len(set(identity.component_ids)) != len(identity.component_ids):
        raise RGCResponseContractError("Synthetic teacher identity components repeat")
    if (
        len(set(zip(identity.cell_group_ids, identity.cell_replicate_ids, strict=True)))
        != cell_count
    ):
        raise RGCResponseContractError("Synthetic teacher identity cell IDs repeat")
    if (
        identity.cells_per_type_polarity < 1
        or cell_count % identity.cells_per_type_polarity != 0
        or not np.isfinite(identity.residual_bound)
        or identity.residual_bound < 0
        or not np.isfinite(identity.context_high_scale).all()
        or not np.isfinite(identity.context_gain_cell_residual).all()
    ):
        raise RGCResponseContractError("Synthetic teacher identity values are invalid")


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


def _read_teacher_identity(handle: h5py.File) -> TeacherIdentityMetadata | None:
    keys = (
        "teacher/revision",
        "teacher/generation_seed",
        "teacher/residual_seed",
        "teacher/cells_per_type_polarity",
        "teacher/residual_bound",
        "teacher/cell_group_id",
        "teacher/cell_replicate_id",
        "teacher/component_id",
        "teacher/context_high_scale",
        "teacher/context_gain_cell_residual",
    )
    present = tuple(key in handle for key in keys)
    if not any(present):
        return None
    if not all(present):
        raise RGCResponseContractError("Synthetic teacher identity metadata is partial")
    return TeacherIdentityMetadata(
        revision=_read_scalar_text(handle, "teacher/revision"),
        generation_seed=_read_int_scalar(handle, "teacher/generation_seed"),
        residual_seed=_read_int_scalar(handle, "teacher/residual_seed"),
        cells_per_type_polarity=_read_int_scalar(
            handle,
            "teacher/cells_per_type_polarity",
        ),
        residual_bound=_read_float_scalar(handle, "teacher/residual_bound"),
        cell_group_ids=_read_vector_text(handle, "teacher/cell_group_id"),
        cell_replicate_ids=_read_vector_text(handle, "teacher/cell_replicate_id"),
        component_ids=_read_vector_text(handle, "teacher/component_id"),
        context_high_scale=_read_float_vector(handle, "teacher/context_high_scale"),
        context_gain_cell_residual=_read_float_vector(
            handle,
            "teacher/context_gain_cell_residual",
        ),
    )


def _required(handle: h5py.File, key: str) -> h5py.Dataset:
    if key not in handle:
        raise RGCResponseContractError(f"Missing required dataset: {key}")
    return handle[key]


def _read_scalar_text(handle: h5py.File, key: str) -> str:
    return _decode_text(_required(handle, key)[()])


def _read_text_vector(handle: h5py.File, key: str) -> tuple[str, ...]:
    values = np.asarray(_required(handle, key)[()]).reshape(-1)
    return tuple(_decode_text(value) for value in values)


def _read_vector_text(handle: h5py.File, key: str) -> tuple[str, ...]:
    values = np.asarray(_required(handle, key)[()])
    if values.ndim != 1:
        raise RGCResponseContractError("Synthetic teacher identity text must be vector")
    return tuple(_decode_text(value) for value in values)


def _read_float_vector(handle: h5py.File, key: str) -> np.ndarray:
    values = np.asarray(_required(handle, key)[()])
    if values.ndim != 1:
        raise RGCResponseContractError(
            "Synthetic teacher identity arrays must be vectors"
        )
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RGCResponseContractError(
            "Synthetic teacher identity arrays must be numeric"
        ) from exc


def _read_int_scalar(handle: h5py.File, key: str) -> int:
    values = np.asarray(_required(handle, key)[()])
    if values.shape not in {(), (1,)} or values.dtype.kind not in {"i", "u"}:
        raise RGCResponseContractError("Integer metadata must be integer scalar")
    return int(values.reshape(-1)[0])


def _read_float_scalar(handle: h5py.File, key: str) -> float:
    values = np.asarray(_required(handle, key)[()])
    if values.shape not in {(), (1,)} or values.dtype.kind not in {"f", "i", "u"}:
        raise RGCResponseContractError("Float metadata must be numeric scalar")
    return float(values.reshape(-1)[0])


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
    "TeacherIdentityMetadata",
    "fingerprint_field_bytes",
    "load_rgc_response",
    "validate_response_splits",
]
