from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from training.response_config import ResponseExperimentConfig


CHECKPOINT_SCHEMA = "retina_rgc_response_snn"
CHECKPOINT_SCHEMA_REVISION = 2


class ResponseCheckpointError(ValueError):
    pass


def save_response_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_nll: float,
    generator: torch.Generator,
    fingerprint: str,
    target_kind: str,
    config: ResponseExperimentConfig,
) -> None:
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_revision": CHECKPOINT_SCHEMA_REVISION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "best_nll": best_nll,
            "sampling_rng": generator.get_state(),
            "dataset_fingerprint": fingerprint,
            "target_kind": target_kind,
            "config": asdict(config),
        },
        Path(path),
    )


def load_response_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None,
    fingerprint: str,
    target_kind: str,
    config: ResponseExperimentConfig,
) -> tuple[int, float]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ResponseCheckpointError("Response checkpoint payload is invalid")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_revision") != CHECKPOINT_SCHEMA_REVISION
    ):
        raise ResponseCheckpointError(
            "Checkpoint is not a response-fitting revision-2 checkpoint; "
            "start a fresh response run"
        )
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ResponseCheckpointError("Response checkpoint dataset fingerprint mismatch")
    if payload.get("target_kind") != target_kind:
        raise ResponseCheckpointError("Response checkpoint target kind mismatch")
    if payload.get("config") != asdict(config):
        raise ResponseCheckpointError("Response checkpoint configuration mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if generator is not None:
        generator.set_state(payload["sampling_rng"])
    return int(payload["optimizer_step"]), float(payload["best_nll"])


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_REVISION",
    "ResponseCheckpointError",
    "load_response_checkpoint",
    "save_response_checkpoint",
]
