from __future__ import annotations

import glob
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from data.rgc_response import (
    CellMetadata,
    RGCResponseContractError,
    RGCResponseSession,
    ResponseTargetKind,
    load_rgc_response,
    validate_response_splits,
)
from training.response_config import ResponseDataConfig


@dataclass(frozen=True, slots=True)
class ResponseSplit:
    cone_response: torch.Tensor
    spike_counts: torch.Tensor
    valid_mask: torch.Tensor
    source_ids: tuple[str, ...]
    context_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedResponseData:
    train: ResponseSplit
    validation: ResponseSplit
    test: ResponseSplit
    cells: CellMetadata
    cone_positions_degs: np.ndarray
    time_axis_seconds: np.ndarray
    target_kind: ResponseTargetKind
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    fingerprint: str

    @property
    def dt_ms(self) -> float:
        return float(np.mean(np.diff(self.time_axis_seconds)) * 1000)


def prepare_response_data(config: ResponseDataConfig) -> PreparedResponseData:
    sessions = (
        _load_glob("train", config.train_glob),
        _load_glob("validation", config.validation_glob),
        _load_glob("test", config.test_glob),
    )
    validate_response_splits(*sessions)
    reference = sessions[0][0]
    for session in (*sessions[0], *sessions[1], *sessions[2]):
        if session.cone_response.shape[1] < config.sequence_steps:
            raise RGCResponseContractError(
                f"{session.path} has fewer than {config.sequence_steps} time bins"
            )
    train_cones = np.concatenate(
        [session.cone_response[:, : config.sequence_steps] for session in sessions[0]]
    )
    mean = train_cones.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_cones.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    split_values = tuple(
        _stack_split(split, config.sequence_steps, mean, std) for split in sessions
    )
    fingerprint = _fingerprint(reference)
    return PreparedResponseData(
        train=split_values[0],
        validation=split_values[1],
        test=split_values[2],
        cells=reference.cells,
        cone_positions_degs=reference.cone_positions_degs,
        time_axis_seconds=reference.time_axis_seconds[: config.sequence_steps],
        target_kind=reference.target_kind,
        normalization_mean=mean,
        normalization_std=std,
        fingerprint=fingerprint,
    )


def sample_response_batch(
    split: ResponseSplit,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stimulus_count, trial_count = split.spike_counts.shape[:2]
    stimulus = torch.randint(
        stimulus_count, (batch_size,), generator=generator
    )
    trial = torch.randint(trial_count, (batch_size,), generator=generator)
    cones = split.cone_response.index_select(0, stimulus)
    counts = split.spike_counts[stimulus, trial]
    mask = split.valid_mask[stimulus, trial]
    return cones.to(device), counts.to(device), mask.to(device)


def _load_glob(name: str, pattern: str) -> tuple[RGCResponseSession, ...]:
    paths = tuple(sorted(glob.glob(pattern)))
    if not paths:
        raise RGCResponseContractError(f"{name} response glob matched no files")
    return tuple(load_rgc_response(path) for path in paths)


def _stack_split(
    sessions: tuple[RGCResponseSession, ...],
    steps: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> ResponseSplit:
    cones = np.concatenate(
        [(session.cone_response[:, :steps] - mean) / std for session in sessions]
    )
    spikes = np.concatenate(
        [session.spike_counts[:, :, :steps] for session in sessions]
    )
    masks = np.concatenate(
        [session.valid_mask[:, :, :steps] for session in sessions]
    )
    return ResponseSplit(
        cone_response=torch.from_numpy(cones.astype(np.float32)),
        spike_counts=torch.from_numpy(spikes.astype(np.float32)),
        valid_mask=torch.from_numpy(masks),
        source_ids=tuple(
            source_id for session in sessions for source_id in session.source_ids
        ),
        context_ids=tuple(
            context_id for session in sessions for context_id in session.context_ids
        ),
    )


def _fingerprint(session: RGCResponseSession) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(session.cells.ids).encode())
    digest.update("\0".join(session.cells.type_ids).encode())
    digest.update(session.cells.polarities.tobytes())
    digest.update(session.cone_positions_degs.tobytes())
    return digest.hexdigest()


__all__ = [
    "PreparedResponseData",
    "ResponseSplit",
    "prepare_response_data",
    "sample_response_batch",
]
