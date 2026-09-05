from __future__ import annotations

from typing import TypedDict

import torch

from models.mechanistic_retina.contracts import (
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)

type ConfigValue = bool | int | float | str | tuple[float, float]


class CheckpointPayload(TypedDict):
    role: str
    model_revision: int
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, ConfigValue]
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]


class DataPayload(TypedDict):
    validation_cones: torch.Tensor
    validation_spikes: torch.Tensor


class ACCircuitInputError(ValueError):
    pass


def rebuild_model(checkpoint: CheckpointPayload) -> MechanisticGraphTemporalRetina:
    if checkpoint.get("role") != "student-trained":
        raise ACCircuitInputError("checkpoint role must be student-trained")
    if checkpoint.get("model_revision") != MECHANISTIC_MODEL_REVISION:
        raise ACCircuitInputError(
            "checkpoint predates bounded-learnable temporal parameters; rerun benchmark"
        )
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(**checkpoint["model_config"]),
        checkpoint["cone_positions"],
        checkpoint["cell_positions"],
        checkpoint["cell_types"],
        checkpoint["polarities"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def validation_tensors(
    data: DataPayload,
    checkpoint: CheckpointPayload,
) -> tuple[torch.Tensor, torch.Tensor]:
    cones = data["validation_cones"]
    spikes = data["validation_spikes"]
    if cones.ndim != 3 or spikes.ndim != 4:
        raise ACCircuitInputError("validation tensors have invalid ranks")
    if cones.shape[0] != spikes.shape[0] or cones.shape[1] != spikes.shape[2]:
        raise ACCircuitInputError("validation cone/spike axes do not match")
    if (
        cones.shape[-1] != checkpoint["cone_positions"].shape[0]
        or spikes.shape[-1] != checkpoint["cell_positions"].shape[0]
    ):
        raise ACCircuitInputError("validation axes do not match checkpoint geometry")
    if not torch.isfinite(cones).all() or not torch.isfinite(spikes).all():
        raise ACCircuitInputError("validation tensors must be finite")
    return cones, spikes


__all__ = [
    "ACCircuitInputError",
    "CheckpointPayload",
    "DataPayload",
    "rebuild_model",
    "validation_tensors",
]
