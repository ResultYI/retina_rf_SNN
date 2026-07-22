from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.config import ExperimentConfig


CHECKPOINT_SCHEMA = "retina_rf_snn"
CHECKPOINT_SCHEMA_REVISION = 1


class CheckpointError(ValueError):
    pass


def checkpoint_payload(
    *,
    optimizer_step: int,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    energy_state: Any,
    augmentation_generator: torch.Generator,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "optimizer_step": optimizer_step,
        "model": model.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "energy_state": asdict(energy_state),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "augmentation": augmentation_generator.get_state(),
        },
        "resolved_config": config.resolved(),
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("Checkpoint schema is incompatible")
    if payload.get("schema_revision") != CHECKPOINT_SCHEMA_REVISION:
        raise CheckpointError("Checkpoint schema revision is incompatible")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_REVISION",
    "CheckpointError",
    "checkpoint_payload",
    "load_checkpoint",
    "save_checkpoint",
]

