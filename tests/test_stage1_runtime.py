from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.hybrid import TrainingStage
from training.stage1_runtime import load_checkpoint, validate_formal_video_paths
from training.stage1_types import TrainStage1Config, TrainStage1Error


def test_checkpoint_loader_requests_weights_only(monkeypatch) -> None:
    # Given
    observed: list[bool] = []

    def fake_load(
        path: Path,
        *,
        map_location: torch.device,
        weights_only: bool,
    ) -> dict[str, object]:
        observed.append(weights_only)
        return {
            "epoch": 0,
            "step": 0,
            "core": {},
            "decoder": {},
            "optimizer": {},
            "stage": "decoder_warmup",
        }

    monkeypatch.setattr(torch, "load", fake_load)

    # When
    load_checkpoint(Path("checkpoint.pt"), torch.device("cpu"))

    # Then
    assert observed == [True]


def test_formal_evidence_requires_held_out_validation() -> None:
    # Given
    config = TrainStage1Config(
        train_h5=(Path("train.h5"),),
        val_h5=(),
        output_dir=Path("run"),
        epochs=1,
        batch_size=1,
        input_steps=2,
        horizons=(1,),
        stage=TrainingStage.DECODER_WARMUP,
        device=torch.device("cpu"),
        seed=7,
        t_bptt=1,
        lr_core=1e-4,
        lr_decoder=1e-3,
        num_workers=0,
        max_clip_fraction=0.01,
        resume=None,
        formal_evidence=True,
    )

    # When / Then
    with pytest.raises(TrainStage1Error, match="held-out validation"):
        validate_formal_video_paths(config)
