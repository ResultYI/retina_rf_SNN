from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import torch

from data.retinal_recording import RealSequenceSplit


_LIVE_START_FRAME: Final = 751
_LIVE_FRAME_COUNT: Final = 90_000
_MOVIE_RATE_HZ: Final = 150.0
_SPIKE_RESOLUTION_MS: Final = 0.1
_FIELD_SIZE_DEG: Final = 4.6
_RECORDING_DURATION_MS: Final = 600_000.0


@dataclass(frozen=True, slots=True)
class SchottdorfDataError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class SpikeTimeTable:
    total_spikes: int
    video_start_ticks: int
    resolution_ms: float
    times_ms: torch.Tensor


@dataclass(frozen=True, slots=True)
class SchottdorfAdapterConfig:
    train_sequence_count: int = 16
    validation_sequence_count: int = 4
    sequence_steps: int = 150
    warmup_steps: int = 30
    crop_pixels: int = 51
    pool_factor: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.train_sequence_count,
            self.validation_sequence_count,
            self.sequence_steps,
            self.crop_pixels,
            self.pool_factor,
        )
        if any(value < 1 for value in positive):
            raise SchottdorfDataError("adapter dimensions must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.sequence_steps:
            raise SchottdorfDataError("warmup must be within each sequence")
        if self.crop_pixels % 2 == 0 or self.crop_pixels % self.pool_factor != 0:
            raise SchottdorfDataError(
                "crop_pixels must be odd and divisible by pool_factor"
            )
        required = (
            self.train_sequence_count + self.validation_sequence_count
        ) * self.sequence_steps
        if required > _LIVE_FRAME_COUNT:
            raise SchottdorfDataError("requested sequences exceed the live movie")


@dataclass(frozen=True, slots=True)
class SchottdorfMacaqueData:
    recording_id: str
    train: RealSequenceSplit
    validation: RealSequenceSplit
    cell_ids: tuple[str, ...]
    recorded_cell_classes: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    cell_positions_degs: torch.Tensor
    cone_positions_degs: torch.Tensor
    dt_ms: float
    stimulus_rate_hz: float
    spike_time_resolution_ms: float
    trial_count: int
    input_representation: str
    crop_pixels: int
    pooled_grid_size: int


def parse_spike_time_table(path: str | Path) -> SpikeTimeTable:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    total_spikes = _header_integer(lines, "Total spikes")
    video_start = _header_integer(lines, "Video Start")
    try:
        table_start = lines.index("No\tTime") + 1
    except ValueError as error:
        raise SchottdorfDataError("spike table lacks No/Time header") from error
    recorded: list[int] = []
    for line in lines[table_start:]:
        fields = line.split("\t")
        if len(fields) != 2 or not fields[1].strip():
            continue
        try:
            recorded.append(int(fields[1]))
        except ValueError as error:
            raise SchottdorfDataError("spike table contains a non-integer time") from error
    if len(recorded) != total_spikes:
        raise SchottdorfDataError("spike table count disagrees with metadata")
    times_ms = torch.tensor(recorded, dtype=torch.float64) * _SPIKE_RESOLUTION_MS
    live = times_ms[(times_ms >= 0) & (times_ms < _RECORDING_DURATION_MS)]
    return SpikeTimeTable(total_spikes, video_start, _SPIKE_RESOLUTION_MS, live)


def load_minimal_macaque_natural_movie(
    recording_dir: str | Path,
    config: SchottdorfAdapterConfig = SchottdorfAdapterConfig(),
) -> SchottdorfMacaqueData:
    directory = Path(recording_dir)
    movie_path = directory / "1x10_256.mpg"
    spike_path = directory / "lSS01300.txt"
    if not movie_path.is_file() or not spike_path.is_file():
        raise SchottdorfDataError("recording lacks movie or lSS01300 spike table")
    total_sequences = config.train_sequence_count + config.validation_sequence_count
    required_frames = total_sequences * config.sequence_steps
    drive, positions = _load_calibrated_lm_drive(movie_path, required_frames, config)
    spike_table = parse_spike_time_table(spike_path)
    counts = _bin_spikes(spike_table.times_ms.numpy(), required_frames)
    sequences = drive.reshape(total_sequences, config.sequence_steps, -1)
    count_sequences = counts.reshape(total_sequences, config.sequence_steps, 1)
    train_stop = config.train_sequence_count
    return SchottdorfMacaqueData(
        recording_id="lSS01300",
        train=_make_split(sequences[:train_stop], count_sequences[:train_stop], config, 0),
        validation=_make_split(
            sequences[train_stop:], count_sequences[train_stop:], config, train_stop
        ),
        cell_ids=("70#34",),
        recorded_cell_classes=("MC on",),
        cell_types=("parasol",),
        polarities=("ON",),
        cell_positions_degs=torch.zeros((1, 2), dtype=torch.float32),
        cone_positions_degs=torch.from_numpy(positions),
        dt_ms=1000.0 / _MOVIE_RATE_HZ,
        stimulus_rate_hz=_MOVIE_RATE_HZ,
        spike_time_resolution_ms=spike_table.resolution_ms,
        trial_count=1,
        input_representation="macaque_experiment_calibrated_l_plus_m_weber_drive_v1",
        crop_pixels=config.crop_pixels,
        pooled_grid_size=config.crop_pixels // config.pool_factor,
    )


def _header_integer(lines: list[str], name: str) -> int:
    prefix = f"{name}\t"
    for line in lines:
        if line.startswith(prefix):
            try:
                return int(line.removeprefix(prefix))
            except ValueError as error:
                raise SchottdorfDataError(f"invalid {name} metadata") from error
    raise SchottdorfDataError(f"spike table lacks {name} metadata")


def _load_calibrated_lm_drive(
    path: Path,
    required_frames: int,
    config: SchottdorfAdapterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SchottdorfDataError("natural-movie stimulus cannot be opened")
    grid_size = config.crop_pixels // config.pool_factor
    blank_sum = np.zeros((grid_size**2,), dtype=np.float64)
    live = np.empty(
        (required_frames, grid_size**2), dtype=np.float32
    )
    try:
        for frame_index in range(_LIVE_START_FRAME + required_frames):
            ok, frame = capture.read()
            if not ok:
                raise SchottdorfDataError("natural-movie stimulus ended early")
            pooled = _pooled_lm_signal(frame, config)
            if frame_index < _LIVE_START_FRAME:
                blank_sum += pooled
            else:
                live[frame_index - _LIVE_START_FRAME] = pooled
    finally:
        capture.release()
    background = blank_sum / _LIVE_START_FRAME
    if not np.isfinite(background).all() or np.any(background <= 0):
        raise SchottdorfDataError("blank-frame cone background is invalid")
    drive = ((live - background) / background).astype(np.float32)
    return drive, _cone_positions(config)


def _pooled_lm_signal(
    frame_bgr: np.ndarray, config: SchottdorfAdapterConfig
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    y0 = (height - config.crop_pixels) // 2
    x0 = (width - config.crop_pixels) // 2
    patch = frame_bgr[
        y0 : y0 + config.crop_pixels,
        x0 : x0 + config.crop_pixels,
        ::-1,
    ].astype(np.float32)
    red, green, blue = np.moveaxis(patch, -1, 0)
    red = 0.01451 + 0.9855 * np.power(red / 256.0, 2.3122)
    green = 0.005123 + 0.9949 * np.power(green / 256.0, 2.2752)
    blue = 0.02612 + 0.9739 * np.power(blue / 256.0, 2.2818)
    l_cone = 2.74 * red + 3.4 * green + 1.34 * blue
    m_cone = 1.21 * (1.06 * red + 3.58 * green + 2.07 * blue)
    size = config.crop_pixels // config.pool_factor
    pooled = (l_cone + m_cone).reshape(
        size, config.pool_factor, size, config.pool_factor
    ).mean((1, 3))
    return pooled.reshape(-1)


def _cone_positions(config: SchottdorfAdapterConfig) -> np.ndarray:
    size = config.crop_pixels // config.pool_factor
    pixel_step = _FIELD_SIZE_DEG / 256.0
    centers = (
        np.arange(size, dtype=np.float32) * config.pool_factor
        + (config.pool_factor - 1) / 2
        - (config.crop_pixels - 1) / 2
    ) * pixel_step
    yy, xx = np.meshgrid(-centers, centers, indexing="ij")
    return np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)


def _bin_spikes(times_ms: np.ndarray, frame_count: int) -> np.ndarray:
    indices = np.floor(times_ms * _MOVIE_RATE_HZ / 1000.0).astype(np.int64)
    selected = indices[(indices >= 0) & (indices < frame_count)]
    return np.bincount(selected, minlength=frame_count).astype(np.int64)


def _make_split(
    drive: np.ndarray,
    counts: np.ndarray,
    config: SchottdorfAdapterConfig,
    sequence_offset: int,
) -> RealSequenceSplit:
    events = (counts > 0).astype(np.float32)
    valid = np.ones_like(events, dtype=bool)
    valid[:, : config.warmup_steps] = False
    sequence_ids = tuple(
        f"live-frames-{(sequence_offset + index) * config.sequence_steps:06d}-"
        f"{(sequence_offset + index + 1) * config.sequence_steps - 1:06d}"
        for index in range(drive.shape[0])
    )
    return RealSequenceSplit(
        cone_drive=torch.from_numpy(drive),
        spike_counts=torch.from_numpy(counts),
        spike_events=torch.from_numpy(events),
        valid_mask=torch.from_numpy(valid),
        source_image_ids=sequence_ids,
        trial_indices=(0,) * drive.shape[0],
    )


__all__ = [
    "SchottdorfAdapterConfig",
    "SchottdorfDataError",
    "SchottdorfMacaqueData",
    "SpikeTimeTable",
    "load_minimal_macaque_natural_movie",
    "parse_spike_time_table",
]
