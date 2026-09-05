from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.retinal_recording import RealSequenceSplit
from data.schottdorf_lee_2021 import (
    SchottdorfAdapterConfig,
    SchottdorfDataError,
    _load_calibrated_lm_drive,
)
from data.schottdorf_lee_catalog import RecordingKind, SchottdorfRecording
from data.schottdorf_lee_spikes import parse_recording_spike_trials


_MOVIE_RATE_HZ = 150.0


@dataclass(frozen=True, slots=True)
class SchottdorfMovieDrive:
    sequences: np.ndarray
    cone_positions_degs: torch.Tensor
    dt_ms: float
    stimulus_rate_hz: float


@dataclass(frozen=True, slots=True)
class SchottdorfCellwiseData:
    recording_ids: tuple[str, ...]
    recording_kinds: tuple[str, ...]
    train: RealSequenceSplit
    validation: RealSequenceSplit
    cell_ids: tuple[str, ...]
    recorded_cell_classes: tuple[str, ...]
    retinal_classes: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    eccentricity_deg: float
    cell_positions_degs: torch.Tensor
    cone_positions_degs: torch.Tensor
    dt_ms: float
    stimulus_rate_hz: float
    spike_time_resolution_ms: float
    trial_count: int
    input_representation: str
    population_locality_constructed: bool

    @property
    def recording_id(self) -> str:
        return self.recording_ids[0]

    @property
    def recording_kind(self) -> str:
        return self.recording_kinds[0]


def load_schottdorf_movie_drive(
    movie_path: str | Path,
    config: SchottdorfAdapterConfig,
) -> SchottdorfMovieDrive:
    total_sequences = config.train_sequence_count + config.validation_sequence_count
    frame_count = total_sequences * config.sequence_steps
    drive, positions = _load_calibrated_lm_drive(Path(movie_path), frame_count, config)
    return SchottdorfMovieDrive(
        sequences=drive.reshape(total_sequences, config.sequence_steps, -1),
        cone_positions_degs=torch.from_numpy(positions),
        dt_ms=1000.0 / _MOVIE_RATE_HZ,
        stimulus_rate_hz=_MOVIE_RATE_HZ,
    )


def load_schottdorf_recording(
    recording: SchottdorfRecording,
    movie: SchottdorfMovieDrive,
    config: SchottdorfAdapterConfig,
) -> SchottdorfCellwiseData:
    spikes = parse_recording_spike_trials(recording)
    total_sequences = config.train_sequence_count + config.validation_sequence_count
    frame_count = total_sequences * config.sequence_steps
    if (
        recording.recording_kind is RecordingKind.REPEATED_ONE_MINUTE
        and frame_count > 9_000
    ):
        raise SchottdorfDataError("6x1min adapter window exceeds the one-minute movie")
    counts = np.stack(
        [
            _bin_trial(times.numpy(), frame_count).reshape(
                total_sequences, config.sequence_steps, 1
            )
            for times in spikes.live_times_ms_by_trial
        ]
    )
    train = _make_trial_split(
        recording.recording_id,
        movie.sequences,
        counts,
        config,
        segment_start=0,
        segment_stop=config.train_sequence_count,
    )
    validation = _make_trial_split(
        recording.recording_id,
        movie.sequences,
        counts,
        config,
        segment_start=config.train_sequence_count,
        segment_stop=total_sequences,
    )
    return SchottdorfCellwiseData(
        recording_ids=(recording.recording_id,),
        recording_kinds=(recording.recording_kind.value,),
        train=train,
        validation=validation,
        cell_ids=(recording.cell_id,),
        recorded_cell_classes=(recording.recorded_cell_class,),
        retinal_classes=(recording.retinal_class,),
        cell_types=(recording.canonical_cell_type,),
        polarities=(recording.polarity,),
        eccentricity_deg=recording.eccentricity_deg,
        cell_positions_degs=torch.zeros((1, 2), dtype=torch.float32),
        cone_positions_degs=movie.cone_positions_degs,
        dt_ms=movie.dt_ms,
        stimulus_rate_hz=movie.stimulus_rate_hz,
        spike_time_resolution_ms=spikes.resolution_ms,
        trial_count=len(spikes.live_times_ms_by_trial),
        input_representation="macaque_experiment_calibrated_l_plus_m_weber_drive_v1",
        population_locality_constructed=False,
    )


def load_schottdorf_cell(
    recordings: tuple[SchottdorfRecording, ...],
    movie: SchottdorfMovieDrive,
    config: SchottdorfAdapterConfig,
) -> SchottdorfCellwiseData:
    if not recordings:
        raise SchottdorfDataError("cell-wise adapter requires at least one recording")
    first = recordings[0]
    if any(
        item.cell_id != first.cell_id
        or item.canonical_cell_type != first.canonical_cell_type
        or item.polarity != first.polarity
        for item in recordings[1:]
    ):
        raise SchottdorfDataError("cell-wise recordings have inconsistent identity")
    datasets = tuple(
        load_schottdorf_recording(recording, movie, config) for recording in recordings
    )
    return SchottdorfCellwiseData(
        recording_ids=tuple(item.recording_id for item in recordings),
        recording_kinds=tuple(item.recording_kind.value for item in recordings),
        train=_concatenate_splits(datasets, validation=False),
        validation=_concatenate_splits(datasets, validation=True),
        cell_ids=(first.cell_id,),
        recorded_cell_classes=tuple(
            dict.fromkeys(item.recorded_cell_class for item in recordings)
        ),
        retinal_classes=(first.retinal_class,),
        cell_types=(first.canonical_cell_type,),
        polarities=(first.polarity,),
        eccentricity_deg=first.eccentricity_deg,
        cell_positions_degs=torch.zeros((1, 2), dtype=torch.float32),
        cone_positions_degs=movie.cone_positions_degs,
        dt_ms=movie.dt_ms,
        stimulus_rate_hz=movie.stimulus_rate_hz,
        spike_time_resolution_ms=datasets[0].spike_time_resolution_ms,
        trial_count=sum(item.trial_count for item in datasets),
        input_representation=datasets[0].input_representation,
        population_locality_constructed=False,
    )


def _bin_trial(times_ms: np.ndarray, frame_count: int) -> np.ndarray:
    indices = np.floor(times_ms * _MOVIE_RATE_HZ / 1000.0).astype(np.int64)
    selected = indices[(indices >= 0) & (indices < frame_count)]
    return np.bincount(selected, minlength=frame_count).astype(np.int64)


def _make_trial_split(
    recording_id: str,
    drive: np.ndarray,
    counts: np.ndarray,
    config: SchottdorfAdapterConfig,
    *,
    segment_start: int,
    segment_stop: int,
) -> RealSequenceSplit:
    trial_count = counts.shape[0]
    pairs = tuple(
        (segment, trial)
        for segment in range(segment_start, segment_stop)
        for trial in range(trial_count)
    )
    split_drive = np.stack([drive[segment] for segment, _ in pairs])
    split_counts = np.stack([counts[trial, segment] for segment, trial in pairs])
    events = (split_counts > 0).astype(np.float32)
    valid = np.ones_like(events, dtype=bool)
    valid[:, : config.warmup_steps] = False
    source_ids = tuple(
        f"{recording_id}-live-frames-{segment * config.sequence_steps:06d}-"
        f"{(segment + 1) * config.sequence_steps - 1:06d}-trial-{trial + 1}"
        for segment, trial in pairs
    )
    return RealSequenceSplit(
        cone_drive=torch.from_numpy(split_drive),
        spike_counts=torch.from_numpy(split_counts),
        spike_events=torch.from_numpy(events),
        valid_mask=torch.from_numpy(valid),
        source_image_ids=source_ids,
        trial_indices=tuple(trial for _, trial in pairs),
    )


def _concatenate_splits(
    datasets: tuple[SchottdorfCellwiseData, ...],
    *,
    validation: bool,
) -> RealSequenceSplit:
    splits = tuple(item.validation if validation else item.train for item in datasets)
    offsets = []
    offset = 0
    for dataset in datasets:
        offsets.append(offset)
        offset += dataset.trial_count
    return RealSequenceSplit(
        cone_drive=torch.cat(tuple(split.cone_drive for split in splits)),
        spike_counts=torch.cat(tuple(split.spike_counts for split in splits)),
        spike_events=torch.cat(tuple(split.spike_events for split in splits)),
        valid_mask=torch.cat(tuple(split.valid_mask for split in splits)),
        source_image_ids=tuple(
            source_id for split in splits for source_id in split.source_image_ids
        ),
        trial_indices=tuple(
            trial + trial_offset
            for split, trial_offset in zip(splits, offsets, strict=True)
            for trial in split.trial_indices
        ),
    )


__all__ = [
    "SchottdorfCellwiseData",
    "SchottdorfMovieDrive",
    "load_schottdorf_cell",
    "load_schottdorf_movie_drive",
    "load_schottdorf_recording",
]
