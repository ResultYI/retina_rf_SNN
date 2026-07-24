from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from evaluation.readout_ladder import (
    ModelReadoutLadderRequest,
    ReadoutLadderResult,
    evaluate_model_readout_ladder,
)
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip
from training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ReadoutReportingRequest:
    initialized_model: RetinaModel
    initialized_decoder: TiedLocalDecoder
    selected_model: RetinaModel
    selected_decoder: TiedLocalDecoder
    training_clips: Sequence[AugmentedClip]
    validation_clips: Sequence[AugmentedClip]
    config: ExperimentConfig
    dt_ms: float
    output_dir: Path


def write_readout_reports(request: ReadoutReportingRequest) -> None:
    common = {
        "training_clips": request.training_clips,
        "validation_clips": request.validation_clips,
        "supervised_steps": request.config.training.supervised_steps,
        "dt_ms": request.dt_ms,
    }
    initialized_ladder = evaluate_model_readout_ladder(
        ModelReadoutLadderRequest(
            model=request.initialized_model,
            gain_max=request.initialized_decoder.gain_max,
            **common,
        )
    )
    selected_ladder = evaluate_model_readout_ladder(
        ModelReadoutLadderRequest(
            model=request.selected_model,
            gain_max=request.selected_decoder.gain_max,
            **common,
        )
    )
    _write_pair(
        request.output_dir / "readout_ladder.json",
        initialized_ladder,
        selected_ladder,
    )


def _write_pair(
    path: Path,
    initialized: ReadoutLadderResult,
    selected: ReadoutLadderResult,
) -> None:
    path.write_text(
        json.dumps(
            {
                "initialized": asdict(initialized),
                "selected": asdict(selected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "ReadoutReportingRequest",
    "write_readout_reports",
]
