"""Read-only bridge from the frozen Schottdorf train split to Population v0.1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.retinal_recording import RealSequenceSplit
from data.schottdorf_lee_2021 import (
    SchottdorfAdapterConfig, _FIELD_SIZE_DEG, _MOVIE_RATE_HZ,
    _load_calibrated_lm_drive,
)
from data.schottdorf_lee_catalog import SchottdorfRecording
from data.schottdorf_lee_multirecording import (
    SchottdorfMovieDrive, _bin_trial, _make_trial_split,
)
from data.schottdorf_lee_spikes import parse_recording_spike_trials
from experiments.retipath_population_v0_1.circuit import Stimulus, Trace


@dataclass(frozen=True)
class RGCMapping:
    instance_id: str
    cell_id: str
    recording_id: str
    recorded_type: str
    output_index: int
    population_port: str
    type_compatible: bool
    blocker: str | None


def recording_mapping(recording: SchottdorfRecording) -> RGCMapping:
    if recording.retinal_class not in {"MC", "PC"} or recording.polarity not in {"ON", "OFF"}:
        raise ValueError("R0 only registers the frozen MC/PC ON/OFF cohort")
    expected_type = "parasol" if recording.retinal_class == "MC" else "midget"
    if recording.canonical_cell_type != expected_type:
        raise ValueError("recording class and canonical type disagree")
    index = 0 if recording.polarity == "ON" else 1
    return RGCMapping(
        instance_id=f"SL21::{recording.cell_id}::{recording.recording_id}",
        cell_id=recording.cell_id, recording_id=recording.recording_id,
        recorded_type=f"{recording.retinal_class} {recording.polarity}",
        output_index=index, population_port=f"probability[..., {index}]",
        type_compatible=recording.retinal_class == "MC",
        blocker=None if recording.retinal_class == "MC" else "PC_MIDGET_FAMILY_NOT_IMPLEMENTED",
    )


def load_training_movie(path: Path, config: SchottdorfAdapterConfig) -> SchottdorfMovieDrive:
    """Reuse calibration exactly, decode only the frozen training movie prefix."""
    frames = config.train_sequence_count * config.sequence_steps
    drive, positions = _load_calibrated_lm_drive(path, frames, config)
    return SchottdorfMovieDrive(
        drive.reshape(config.train_sequence_count, config.sequence_steps, -1),
        torch.from_numpy(positions), 1000.0 / _MOVIE_RATE_HZ, _MOVIE_RATE_HZ,
    )


def load_recording_train(
    recording: SchottdorfRecording, movie: SchottdorfMovieDrive,
    config: SchottdorfAdapterConfig,
) -> tuple[RealSequenceSplit, dict]:
    # The existing parser reads the source file; only train-window events are binned.
    parsed = parse_recording_spike_trials(recording)
    frames = config.train_sequence_count * config.sequence_steps
    if frames > 9000:
        raise ValueError("R0 must stay within the existing common one-minute support")
    counts = np.stack([
        _bin_trial(times.numpy(), frames).reshape(config.train_sequence_count, config.sequence_steps, 1)
        for times in parsed.live_times_ms_by_trial
    ])
    split = _make_trial_split(
        recording.recording_id, movie.sequences, counts, config,
        segment_start=0, segment_stop=config.train_sequence_count,
    )
    return split, {
        "biological_trials": len(parsed.live_times_ms_by_trial),
        "resolution_ms": parsed.resolution_ms,
        "video_start_ticks_not_subtracted": parsed.video_start_ticks,
        "duplicate_payload_removed_by_existing_parser": parsed.duplicate_payload_removed,
    }


def physical_stimulus(
    values: torch.Tensor, centers_deg: torch.Tensor,
    config: SchottdorfAdapterConfig, *, dtype: torch.dtype = torch.float64,
) -> Stimulus:
    """Attach physical pixel boxes; no image resize, re-centering or degree rescale."""
    grid = config.crop_pixels // config.pool_factor
    if values.ndim != 3 or values.shape[-1] != grid * grid:
        raise ValueError("unexpected source pixel grid")
    if centers_deg.shape != (grid * grid, 2):
        raise ValueError("source pixel center count differs from stimulus")
    centers = centers_deg.to(dtype=dtype, device=values.device)
    edges = (torch.arange(grid + 1, dtype=dtype, device=values.device)
             * config.pool_factor - config.crop_pixels / 2) * (_FIELD_SIZE_DEG / 256.0)
    row, column = torch.meshgrid(torch.arange(grid, device=values.device),
                                 torch.arange(grid, device=values.device), indexing="ij")
    bounds = torch.stack((torch.stack((edges[column], edges[column + 1]), -1),
                          torch.stack((edges[grid - 1 - row], edges[grid - row]), -1)), -2).reshape(-1, 2, 2)
    torch.testing.assert_close(bounds.mean(-1), centers, rtol=0, atol=torch.finfo(centers_deg.dtype).eps)
    area = (bounds[..., 1] - bounds[..., 0]).prod(-1)
    x = values.to(dtype=dtype)
    result = Stimulus(
        x, bounds, area,
        (torch.arange(x.shape[1], dtype=dtype, device=x.device) + .5) * (1000 / _MOVIE_RATE_HZ),
        torch.ones_like(x, dtype=torch.bool),
    )
    result.validate()
    return result


def history_events(target: torch.Tensor, mapping: RGCMapping, *, dtype: torch.dtype) -> torch.Tensor:
    if target.ndim != 3 or target.shape[-1] != 1 or not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("recorded target must be binary [batch,time,1]")
    events = torch.zeros((*target.shape[:2], 2), dtype=dtype, device=target.device)
    events[..., mapping.output_index] = target[..., 0]
    return events


def recorded_probability(trace: Trace, mapping: RGCMapping) -> torch.Tensor:
    # The other output is unobserved, never a negative target or another biological cell.
    return trace.outputs["probability"][..., mapping.output_index:mapping.output_index + 1]
