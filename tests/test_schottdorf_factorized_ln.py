from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829"


def test_given_ln_when_resolving_fairness_adapter_then_it_exists() -> None:
    assert importlib.util.find_spec("evaluation.mechanistic_retina.schottdorf_ln_source") is not None


def test_given_frozen_cell_when_loading_then_native_data_contract_is_identical() -> None:
    module = importlib.import_module("evaluation.mechanistic_retina.schottdorf_ln_source")
    paths = module.LNSourcePaths(
        ROOT / "data/real/schottdorf_lee_2021_repository",
        ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg", SOURCE,
    )
    checkpoint = SOURCE / "cells/67_4/model-trained.pt"
    before = sha256_file(checkpoint)
    loaded = module.load_ln_cell(paths, "67#4")
    data = loaded.data
    assert data.train.cone_drive.shape == (16, 150, 289)
    assert data.validation.cone_drive.shape == (4, 150, 289)
    assert int(data.train.valid_mask.sum()) == 1920
    assert int(data.validation.valid_mask.sum()) == 480
    assert data.recording_ids == ("lSS01071",)
    assert data.dt_ms == 1000 / 150
    assert loaded.history.tau_ms == 30
    assert torch.equal(data.train.spike_events, (data.train.spike_counts > 0).float())
    assert sha256_file(checkpoint) == before


def test_given_source_output_overlap_when_running_then_reject_before_reading_or_training() -> None:
    module = importlib.import_module("evaluation.mechanistic_retina.schottdorf_center_surround_ln")
    source_module = importlib.import_module("evaluation.mechanistic_retina.schottdorf_ln_source")
    paths = source_module.LNSourcePaths(ROOT / "missing", ROOT / "missing.mpg", SOURCE)
    with pytest.raises(ValueError, match="outside"):
        module.run_ln_cell(paths, "67#4", SOURCE / "forbidden-ln")
