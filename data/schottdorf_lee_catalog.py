from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final


@unique
class RecordingKind(StrEnum):
    REPEATED_ONE_MINUTE = "6x1min"
    TEN_MINUTE = "10min"


@dataclass(frozen=True, slots=True)
class SchottdorfRecording:
    recording_id: str
    path: Path
    cell_id: str
    recorded_cell_class: str
    retinal_class: str
    canonical_cell_type: str
    polarity: str
    recording_kind: RecordingKind
    catalog_recording_kind: RecordingKind
    eccentricity_deg: float


_CLASS_MAPPING: Final = {
    "MC on": ("MC", "parasol", "ON"),
    "MC off": ("MC", "parasol", "OFF"),
    "M on": ("MC", "parasol", "ON"),
    "M off": ("MC", "parasol", "OFF"),
    "+L-Mon": ("PC", "midget", "ON"),
    "+M-Lon": ("PC", "midget", "ON"),
    "-M+Loff": ("PC", "midget", "OFF"),
    "+M-Loff": ("PC", "midget", "OFF"),
    "-L+Moff": ("PC", "midget", "OFF"),
    "S on": ("S", "small-bistratified", "ON"),
}

_RAW_KIND_OVERRIDES: Final = {
    "lSS01184": RecordingKind.REPEATED_ONE_MINUTE,
}

_CATALOG_ROWS: Final = (
    ("lSS01071", "-M+Loff", "10min", "67#4", 6.89),
    ("lSS01078", "MC off", "6x1min", "67#6", 6.72),
    ("lSS01079", "MC off", "10min", "67#6", 6.72),
    ("lSS01086", "MC on", "6x1min", "67#7", 7.36),
    ("lSS01087", "MC on", "10min", "67#7", 7.36),
    ("lSS01110", "+L-Mon", "6x1min", "67#14", 8.31),
    ("lSS01112", "+L-Mon", "10min", "67#14", 8.31),
    ("lSS01130", "+M-Lon", "6x1min", "67#21", 7.85),
    ("lSS01131", "+M-Lon", "10min", "67#21", 7.85),
    ("lSS01141", "+M-Lon", "6x1min", "67#26", 5.83),
    ("lSS01142", "+M-Lon", "10min", "67#26", 5.83),
    ("lSS01159", "MC off", "6x1min", "67#33", 4.73),
    ("lSS01160", "MC off", "10min", "67#33", 4.73),
    ("lSS01167", "+L-Mon", "6x1min", "67#34", 4.80),
    ("lSS01168", "+L-Mon", "10min", "67#34", 4.80),
    ("lSS01178", "S on", "6x1min", "68#1", 5.63),
    ("lSS01180", "S on", "10min", "68#1", 5.63),
    ("lSS01181", "MC off", "6x1min", "68#3", 5.08),
    ("lSS01183", "MC off", "10min", "68#3", 5.08),
    ("lSS01184", "+L-Mon", "10min", "68#4", 5.00),
    ("lSS01194", "+L-Mon", "6x1min", "68#7", 5.72),
    ("lSS01196", "+L-Mon", "10min", "68#7", 5.72),
    ("lSS01221", "MC on", "10min", "68#10", 4.89),
    ("lSS01225", "-M+Loff", "6x1min", "68#11", 5.00),
    ("lSS01227", "-M+Loff", "10min", "68#11", 5.00),
    ("lSS01229", "S on", "6x1min", "68#12", 4.91),
    ("lSS01231", "S on", "10min", "68#12", 4.91),
    ("lSS01251", "+M-Loff", "6x1min", "69#3", 4.59),
    ("lSS01252", "+M-Loff", "10min", "69#3", 4.59),
    ("lSS01254", "MC on", "6x1min", "69#4", 13.00),
    ("lSS01256", "M off", "6x1min", "69#6", 3.88),
    ("lSS01257", "M off", "10min", "69#6", 3.88),
    ("lSS01258", "M on", "6x1min", "69#7", 3.52),
    ("lSS01259", "M on", "10min", "69#7", 3.52),
    ("lSS01270", "-L+Moff", "6x1min", "69#21", 4.40),
    ("lSS01278", "+M-Lon", "6x1min", "70#1", 4.56),
    ("lSS01284", "+M-Lon", "6x1min", "70#7", 3.49),
    ("lSS01285", "+M-Lon", "10min", "70#7", 3.49),
    ("lSS01287", "+M-Lon", "6x1min", "70#15", 3.78),
    ("lSS01299", "MC on", "6x1min", "70#34", 5.66),
    ("lSS01300", "MC on", "10min", "70#34", 5.66),
    ("lSS01302", "S on", "6x1min", "71#9", 4.58),
    ("lSS01303", "S on", "10min", "71#9", 4.58),
)


def public_recordings(recording_dir: str | Path) -> tuple[SchottdorfRecording, ...]:
    directory = Path(recording_dir)
    recordings = []
    for recording_id, raw_class, raw_kind, cell_id, eccentricity in _CATALOG_ROWS:
        retinal_class, canonical_type, polarity = _CLASS_MAPPING[raw_class]
        catalog_kind = RecordingKind(raw_kind)
        recordings.append(
            SchottdorfRecording(
                recording_id=recording_id,
                path=directory / f"{recording_id}.txt",
                cell_id=cell_id,
                recorded_cell_class=raw_class,
                retinal_class=retinal_class,
                canonical_cell_type=canonical_type,
                polarity=polarity,
                recording_kind=_RAW_KIND_OVERRIDES.get(recording_id, catalog_kind),
                catalog_recording_kind=catalog_kind,
                eccentricity_deg=eccentricity,
            )
        )
    return tuple(recordings)


def mc_pc_recordings(recording_dir: str | Path) -> tuple[SchottdorfRecording, ...]:
    recordings = tuple(
        recording
        for recording in public_recordings(recording_dir)
        if recording.retinal_class in {"MC", "PC"}
    )
    missing = tuple(
        str(recording.path) for recording in recordings if not recording.path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"public spike recordings are missing: {missing}")
    return recordings


__all__ = [
    "RecordingKind",
    "SchottdorfRecording",
    "mc_pc_recordings",
    "public_recordings",
]
