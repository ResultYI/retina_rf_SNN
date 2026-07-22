from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from evaluation.dynamic_rf import build_matched_context_pairs
from training.config import load_config
from training.data import PreparedClip


ROOT = Path(__file__).resolve().parents[1]


def test_matched_context_pair_keeps_identical_final_probe() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    evaluation = replace(
        config.evaluation,
        dynamic_rf_max_sources=1,
        dynamic_rf_lag_steps=8,
    )
    clean = torch.arange(config.data.sequence_steps * 3, dtype=torch.float32).reshape(
        config.data.sequence_steps, 3
    )
    clip = PreparedClip(clean=clean, source_id="source-a")
    pair = build_matched_context_pairs([clip], config.data, evaluation)[0]
    assert torch.equal(pair.final_probe, clean[-8:])
    assert torch.equal(
        pair.low_context,
        clean[:-8] * config.data.context_gain_min,
    )
    assert torch.equal(
        pair.high_context,
        clean[:-8] * config.data.context_gain_max,
    )
    assert pair.source_id == "source-a"
