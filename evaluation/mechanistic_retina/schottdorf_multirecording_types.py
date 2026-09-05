from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig


@dataclass(frozen=True, slots=True)
class SchottdorfMultiRunConfig:
    repository_dir: Path
    movie_path: Path
    output_dir: Path
    recording_ids: tuple[str, ...] | None = None
    steps: int = 50
    learning_rate: float = 0.03
    batch_size: int = 4
    seed: int = 202_608_28
    adapter: SchottdorfAdapterConfig = SchottdorfAdapterConfig()


@dataclass(frozen=True, slots=True)
class SchottdorfMultiRunResult:
    artifact_dir: Path
    recording_count: int
    cell_count: int
    stable_cell_count: int
    improved_cell_count: int


@dataclass(frozen=True, slots=True)
class SchottdorfMultiRunError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class SourceLineageError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


class RecordingMetadata(TypedDict):
    recording_id: str
    recorded_cell_class: str
    retinal_class: str
    canonical_cell_type: str
    polarity: str
    recording_kind: str
    catalog_recording_kind: str
    eccentricity_deg: float
    spike_sha256: str


class CellTrainingSummary(TypedDict):
    seed: int
    gradients_finite: bool
    actually_updated: tuple[str, ...]
    major_parameter_groups_updated: dict[str, bool]
    self_edge_connection_parameter_updated: bool


class CellFitRecord(TypedDict):
    recording_ids: tuple[str, ...]
    recording_kinds: tuple[str, ...]
    recording_count: int
    recording_metadata: list[RecordingMetadata]
    input_representation: str
    cell_id: str
    recorded_cell_classes: tuple[str, ...]
    retinal_class: str
    canonical_cell_type: str
    polarity: str
    eccentricity_deg: float
    native_dt_ms: float
    stimulus_rate_hz: float
    spike_time_resolution_ms: float
    biological_trials: int
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    train_event_rate: float
    validation_event_rate: float
    train_multi_spike_bin_fraction: float
    validation_multi_spike_bin_fraction: float
    time_segment_disjoint: bool
    validation_nll_raw: float
    validation_nll_trained: float
    nll_improvement: float
    prediction_improved: bool
    training: CellTrainingSummary
    source_sha256: dict[str, str]


class GroupSummaryRow(TypedDict):
    recordings: int
    cells: int
    mean_nll_raw: float | None
    mean_nll_trained: float | None
    improved_cells: int


class TensorSummary(TypedDict):
    values: list[float]
    minimum: float
    maximum: float
    mean: float
    norm: float


__all__ = [
    "CellFitRecord",
    "CellTrainingSummary",
    "GroupSummaryRow",
    "RecordingMetadata",
    "SchottdorfMultiRunConfig",
    "SchottdorfMultiRunError",
    "SchottdorfMultiRunResult",
    "SourceLineageError",
    "TensorSummary",
]
