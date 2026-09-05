from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

from baselines.center_surround_ln import LNError
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import (
    SchottdorfCellwiseData, load_schottdorf_cell, load_schottdorf_movie_drive,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from training.mechanistic_retina.center_surround_ln import LNHistory


@dataclass(frozen=True, slots=True)
class LNSourcePaths:
    repository: Path
    movie: Path
    retinal_artifact: Path


class SourceCell(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    recording_ids: tuple[str, ...]
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    native_dt_ms: float
    source_sha256: dict[str, str]


class FrozenSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_name: Literal["schottdorf_lee_2021_macaque_cellwise_canonical_v1"] = Field(alias="schema")
    revision: Literal[4] = Field(alias="model_revision")
    cell_count: Literal[22]
    recording_count: Literal[37]
    adapter_config: SchottdorfAdapterConfig
    source_sha256: dict[str, str]
    input_representation: str
    cells: tuple[SourceCell, ...]


@dataclass(frozen=True, slots=True)
class LoadedLNCell:
    data: SchottdorfCellwiseData
    history: LNHistory
    adapter: SchottdorfAdapterConfig
    source_hashes: dict[str, str]


def load_ln_cell(paths: LNSourcePaths, cell_id: str) -> LoadedLNCell:
    results_path = paths.retinal_artifact / "results.json"
    hashes = {str(results_path.resolve()): sha256_file(results_path)}
    source = FrozenSource.model_validate_json(results_path.read_text(encoding="utf-8"))
    cells = {cell.cell_id: cell for cell in source.cells}
    if cell_id not in cells or len(cells) != 22:
        raise LNError("cell must belong to the frozen 22-cell protocol")
    expected = cells[cell_id]
    recordings = tuple(r for r in mc_pc_recordings(paths.repository / "data") if r.cell_id == cell_id)
    if tuple(r.recording_id for r in recordings) != expected.recording_ids:
        raise LNError("recording identities differ from the frozen protocol")
    for recording in recordings:
        digest = sha256_file(recording.path)
        if digest != expected.source_sha256[recording.path.name]:
            raise LNError("spike source hash mismatch")
        hashes[str(recording.path.resolve())] = digest
    movie_hash = sha256_file(paths.movie)
    if movie_hash != source.source_sha256[paths.movie.name]:
        raise LNError("movie source hash mismatch")
    hashes[str(paths.movie.resolve())] = movie_hash
    adapter = source.adapter_config
    if adapter.crop_pixels // adapter.pool_factor != 17:
        raise LNError("LN requires the native 17x17 pooled input")
    movie = load_schottdorf_movie_drive(paths.movie, adapter)
    data = load_schottdorf_cell(recordings, movie, adapter)
    current = (data.train.cone_drive.shape[0], data.validation.cone_drive.shape[0],
               int(data.train.valid_mask.sum()), int(data.validation.valid_mask.sum()), data.dt_ms)
    declared = (expected.train_sequences, expected.validation_sequences,
                expected.train_valid_bins, expected.validation_valid_bins, expected.native_dt_ms)
    if current != declared or data.input_representation != source.input_representation:
        raise LNError("native data, split or valid-bin contract mismatch")
    if set(data.train.source_image_ids) & set(data.validation.source_image_ids):
        raise LNError("original training and validation segments overlap")
    checkpoint_path = paths.retinal_artifact / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
    hashes[str(checkpoint_path.resolve())] = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["revision"] != 4 or checkpoint["cell_id"] != cell_id:
        raise LNError("Canonical history metadata checkpoint mismatch")
    if float(checkpoint["model_config"]["dt_ms"]) != data.dt_ms:
        raise LNError("Canonical and LN native dt differ")
    history = LNHistory(data.dt_ms, float(checkpoint["model_config"]["history_tau_ms"]))
    return LoadedLNCell(data, history, adapter, hashes)
