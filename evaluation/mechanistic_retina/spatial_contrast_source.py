from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from baselines.center_surround_ln import CenterSurroundLN, LNError
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import SchottdorfMovieDrive, load_schottdorf_movie_drive
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


class CellSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    retinal_class: Literal["MC", "PC"]
    polarity: Literal["ON", "OFF"]
    recording_ids: tuple[str, ...]
    native_dt_ms: float
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    validation_nll_trained: float
    parameter_counts: dict[str, int]


class PopulationSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    causal_contract: Literal["h1-shared-bc-direct-broad-ac"]
    cell_count: Literal[22]
    recording_count: Literal[37]
    adapter_config: SchottdorfAdapterConfig
    cells: tuple[CellSource, ...]


class SourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_sha256: dict[str, str]
    cell_ids: tuple[str, ...]


class HistoryMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    dt_ms: float
    tau_ms: float


class LNCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    schema_name: Literal["schottdorf_center_surround_separable_ln_v1"] = Field(alias="schema")
    cell_id: str
    recording_ids: tuple[str, ...]
    context_bins: Literal[60]
    seed: int
    selected_lambda: float
    best_step: int
    stop_step: int
    refit_steps: int
    history: HistoryMetadata
    source_hashes: dict[str, str]
    state: dict[str, torch.Tensor] = Field(alias="model")


class Predictions(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    target: torch.Tensor
    valid_mask: torch.Tensor
    source_image_ids: tuple[str, ...]
    trial_indices: tuple[int, ...]
    logits_trained: torch.Tensor


@dataclass(frozen=True, slots=True)
class Sources:
    population: PopulationSource
    movie: SchottdorfMovieDrive
    recordings: tuple[SchottdorfRecording, ...]
    hashes: dict[str, str]
    ln_root: Path
    retinal_root: Path


def load_sources(root: Path) -> Sources:
    retinal = root / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
    ln = root / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830"
    population = PopulationSource.model_validate_json((retinal / "results.json").read_text())
    manifest = SourceManifest.model_validate_json((retinal / "run-manifest.json").read_text())
    if tuple(c.cell_id for c in population.cells) != manifest.cell_ids or len(population.cells) != 22:
        raise LNError("STOP: final 22-cell manifest identity mismatch")
    config = population.adapter_config
    if (config.train_sequence_count, config.validation_sequence_count, config.sequence_steps,
            config.warmup_steps, config.crop_pixels, config.pool_factor) != (16, 4, 150, 30, 51, 3):
        raise LNError("STOP: final adapter contract differs")
    recordings = mc_pc_recordings(root / "data/real/schottdorf_lee_2021_repository/data")
    expected_recordings = {r for c in population.cells for r in c.recording_ids}
    if len(expected_recordings) != 37 or expected_recordings != {r.recording_id for r in recordings}:
        raise LNError("STOP: final 37-recording identity mismatch")
    movie_path = root / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
    required = [root / p for p in (
        "data/retinal_recording.py", "data/schottdorf_lee_2021.py", "data/schottdorf_lee_catalog.py",
        "data/schottdorf_lee_multirecording.py", "data/schottdorf_lee_spikes.py",
        "training/mechanistic_retina/r4_development.py", "training/mechanistic_retina/losses.py",
        "training/mechanistic_retina/real_sampled.py")]
    expected_hashes = {Path(p).resolve(): h for p, h in manifest.source_sha256.items()}
    hashes = {}
    for path in [*required, movie_path, *[r.path for r in recordings]]:
        digest = sha256_file(path)
        if digest != expected_hashes[path.resolve()]:
            raise LNError(f"STOP: final data/loss source hash differs: {path}")
        hashes[str(path)] = digest
    for path in (retinal / "results.json", retinal / "run-manifest.json", ln / "results.json",
                 root / "baselines/center_surround_ln.py"):
        hashes[str(path)] = sha256_file(path)
    movie = load_schottdorf_movie_drive(movie_path, config)
    return Sources(population, movie, recordings, hashes, ln, retinal)


def load_center_source(sources: Sources, cell: CellSource) -> tuple[CenterSurroundLN, LNCheckpoint]:
    path = sources.ln_root / "cells" / cell.cell_id.replace("#", "_") / "ln-trained.pt"
    checkpoint = LNCheckpoint.model_validate(torch.load(path, map_location="cpu", weights_only=True))
    if checkpoint.cell_id != cell.cell_id or checkpoint.recording_ids != cell.recording_ids:
        raise LNError("STOP: LN checkpoint cell/recording identity mismatch")
    if checkpoint.history.dt_ms != cell.native_dt_ms or checkpoint.refit_steps != checkpoint.best_step:
        raise LNError("STOP: LN refit/time metadata mismatch")
    sources.hashes[str(path)] = sha256_file(path)
    for filename, digest in checkpoint.source_hashes.items():
        if sha256_file(Path(filename)) != digest:
            raise LNError(f"STOP: LN source provenance mismatch: {filename}")
        sources.hashes[filename] = digest
    model = CenterSurroundLN(checkpoint.history.dt_ms, checkpoint.history.tau_ms, checkpoint.seed)
    model.load_state_dict(checkpoint.state, strict=True)
    model.eval().requires_grad_(False)
    if any(not torch.equal(value, checkpoint.state[name]) for name, value in model.state_dict().items()):
        raise LNError("STOP: LN state failed exact reconstruction")
    return model, checkpoint


def compare_saved_predictions(path: Path, split: RealSequenceSplit) -> Predictions:
    saved = Predictions.model_validate(torch.load(path, map_location="cpu", weights_only=True))
    if not torch.equal(saved.target, split.spike_events) or not torch.equal(saved.valid_mask, split.valid_mask):
        raise LNError("STOP: saved production target or exact loss mask mismatch")
    if saved.source_image_ids != split.source_image_ids or saved.trial_indices != split.trial_indices:
        raise LNError("STOP: saved sequence/trial order mismatch")
    return saved


def prediction_nll(saved: Predictions) -> float:
    return spike_prediction_metrics(saved.logits_trained, saved.target, saved.valid_mask).population_nll


def verify_split(split: RealSequenceSplit, expected_bins: int) -> None:
    mask = np.broadcast_to((np.arange(150) >= 30)[None, :, None], split.valid_mask.shape)
    if not np.array_equal(split.valid_mask.numpy(), mask) or int(split.valid_mask.sum()) != expected_bins:
        raise LNError("STOP: exact loss mask differs")
    if not torch.equal((split.spike_counts > 0).to(split.spike_events), split.spike_events):
        raise LNError("STOP: Bernoulli target differs from production counts")
