from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from evaluation.mechanistic_retina.clean_sampled_data import CleanBenchmarkState
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
)
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def save_clean_checkpoint(
    path: Path,
    model: MechanisticGraphTemporalRetina,
    state: CleanBenchmarkState,
    role: str,
) -> None:
    model_config = asdict(model.config)
    model_config["architecture_mode"] = ArchitectureMode(
        model.config.architecture_mode
    ).value
    torch.save(
        {
            "role": role,
            "model_revision": MECHANISTIC_MODEL_REVISION,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "config": asdict(state.config),
            "cone_positions": state.cone_positions,
            "cell_positions": state.cell_positions,
            "cell_types": state.cell_types,
            "polarities": state.polarities,
        },
        path,
    )


__all__ = ["save_clean_checkpoint"]
