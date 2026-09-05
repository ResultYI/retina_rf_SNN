#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: imported by run.py in D:/anaconda/python.exe -B.
# Preserve the existing numerical environment; no dependency resolution.
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
SOURCE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
APPLICATION: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830"
ORIGINAL: Final = ROOT / "output/real_data/schottdorf_r4_dev_visual_illusions_20260830"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APPLICATION))

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import SchottdorfCellwiseData, SchottdorfMovieDrive, load_schottdorf_cell
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina


class Cell(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    recording_ids: tuple[str, ...]
    retinal_class: str
    polarity: str
    validation_nll_trained: float
    full_train_nll_raw: float
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    configuration: MechanisticRetinaConfig = Field(alias="model_config")
    training_contract: dict[str, JsonValue]
    inner_boundaries: tuple[dict[str, int], ...]
    best_step: int
    stopping_step: int

    @property
    def group(self) -> str:
        return f"{self.retinal_class} {self.polarity}"

    @property
    def primary_seed(self) -> int:
        return int(str(self.training_contract["seed"]))

    @property
    def directory(self) -> Path:
        return SOURCE / "cells" / self.cell_id.replace("#", "_")


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_count: int
    recording_count: int
    cells: tuple[Cell, ...]
    adapter_config: SchottdorfAdapterConfig


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_data(cell: Cell, movie: SchottdorfMovieDrive, adapter: SchottdorfAdapterConfig) -> SchottdorfCellwiseData:
    catalog = mc_pc_recordings(ROOT / "data/real/schottdorf_lee_2021_repository/data")
    records = tuple(r for r in catalog if r.cell_id == cell.cell_id)
    assert tuple(r.recording_id for r in records) == cell.recording_ids
    data = load_schottdorf_cell(records, movie, adapter)
    assert (len(data.train.source_image_ids), len(data.validation.source_image_ids),
            int(data.train.valid_mask.sum()), int(data.validation.valid_mask.sum())) == (
                cell.train_sequences, cell.validation_sequences, cell.train_valid_bins, cell.validation_valid_bins)
    saved = torch.load(cell.directory / "validation-predictions.pt", weights_only=True)
    assert torch.equal(data.validation.spike_events, saved["target"])
    assert torch.equal(data.validation.valid_mask, saved["valid_mask"])
    assert data.validation.source_image_ids == tuple(saved["source_image_ids"])
    assert data.validation.trial_indices == tuple(saved["trial_indices"])
    assert tuple(asdict(b) for b in make_inner_dev(data.train).boundaries) == cell.inner_boundaries
    assert data.dt_ms == cell.configuration.dt_ms == 1000 / 150
    return data


def fresh(cell: Cell, data: SchottdorfCellwiseData, seed: int) -> MechanisticGraphTemporalRetina:
    torch.manual_seed(seed)
    return build_mechanistic_retina(cell.configuration, data.cone_positions_degs,
        data.cell_positions_degs, data.cell_types, data.polarities)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
