from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import pickle

import torch
from torch import nn

from training.response_config import ResponseExperimentConfig


CHECKPOINT_SCHEMA = "retina_rgc_response_snn"
CHECKPOINT_SCHEMA_REVISION = 4
MODEL_CONTRACT_REVISION = 2
CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "schema_revision",
        "model",
        "optimizer",
        "optimizer_step",
        "best_nll",
        "best_checkpoint_step",
        "sampling_rng",
        "dataset_fingerprint",
        "target_kind",
        "type_prior_sha256",
        "config",
        "run_id",
        "parent_run_id",
        "model_contract_revision",
        "checkpoint_kind",
    }
)


class ResponseCheckpointError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseCheckpointState:
    optimizer_step: int
    best_nll: float
    best_checkpoint_step: int
    run_id: str
    parent_run_id: str | None
    checkpoint_kind: str


def save_response_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_nll: float,
    best_checkpoint_step: int,
    generator: torch.Generator,
    fingerprint: str,
    target_kind: str,
    config: ResponseExperimentConfig,
    run_id: str,
    checkpoint_kind: str,
    parent_run_id: str | None = None,
) -> None:
    if not run_id or checkpoint_kind not in {"best", "last"}:
        raise ResponseCheckpointError("Checkpoint lineage and kind must be explicit")
    if best_checkpoint_step < 0 or best_checkpoint_step > optimizer_step:
        raise ResponseCheckpointError("best_checkpoint_step is invalid")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_step": optimizer_step,
        "best_nll": best_nll,
        "best_checkpoint_step": best_checkpoint_step,
        "sampling_rng": generator.get_state(),
        "dataset_fingerprint": fingerprint,
        "target_kind": target_kind,
        "type_prior_sha256": _type_prior_sha256(config),
        "config": asdict(config),
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "model_contract_revision": MODEL_CONTRACT_REVISION,
        "checkpoint_kind": checkpoint_kind,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_response_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None,
    fingerprint: str,
    target_kind: str,
    config: ResponseExperimentConfig,
    expected_run_id: str | None = None,
) -> ResponseCheckpointState:
    payload = _validated_checkpoint_payload(Path(path))
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ResponseCheckpointError("Response checkpoint dataset fingerprint mismatch")
    if payload.get("target_kind") != target_kind:
        raise ResponseCheckpointError("Response checkpoint target kind mismatch")
    config_values = asdict(config)
    accepted_configs = [config_values]
    if config.training.supervised_tail_steps is None:
        accepted_configs.append(
            {
                **config_values,
                "training": {
                    key: value
                    for key, value in config_values["training"].items()
                    if key != "supervised_tail_steps"
                },
            }
        )
    if (
        payload.get("config") not in accepted_configs
        or payload.get("type_prior_sha256") != _type_prior_sha256(config)
    ):
        raise ResponseCheckpointError("Response checkpoint configuration mismatch")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ResponseCheckpointError("Response checkpoint run lineage mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if generator is not None:
        generator.set_state(payload["sampling_rng"])
    return _checkpoint_state(payload)


def inspect_response_checkpoint(path: str | Path) -> ResponseCheckpointState:
    return _checkpoint_state(_validated_checkpoint_payload(Path(path)))


def _validated_checkpoint_payload(path: Path) -> Mapping:
    payload = _load_safe_checkpoint_payload(path)
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_revision") != CHECKPOINT_SCHEMA_REVISION
        or payload.get("model_contract_revision") != MODEL_CONTRACT_REVISION
    ):
        raise ResponseCheckpointError(
            "Checkpoint is not a response-fitting revision-4 checkpoint; "
            "start a fresh response run"
        )
    if (
        not isinstance(payload["run_id"], str)
        or not payload["run_id"]
        or payload["checkpoint_kind"] not in {"best", "last"}
    ):
        raise ResponseCheckpointError("Response checkpoint lineage is invalid")
    return payload


def _checkpoint_state(payload: Mapping) -> ResponseCheckpointState:
    return ResponseCheckpointState(
        optimizer_step=int(payload["optimizer_step"]),
        best_nll=float(payload["best_nll"]),
        best_checkpoint_step=int(payload["best_checkpoint_step"]),
        run_id=str(payload["run_id"]),
        parent_run_id=(
            None
            if payload["parent_run_id"] is None
            else str(payload["parent_run_id"])
        ),
        checkpoint_kind=str(payload["checkpoint_kind"]),
    )


def _load_safe_checkpoint_payload(path: Path) -> Mapping:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ResponseCheckpointError(
            "Response checkpoint contains unsafe pickle content"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != CHECKPOINT_KEYS
        or not _is_safe_checkpoint_value(payload)
        or not isinstance(payload["model"], Mapping)
        or not isinstance(payload["optimizer"], Mapping)
        or not isinstance(payload["sampling_rng"], torch.Tensor)
    ):
        raise ResponseCheckpointError("Response checkpoint payload is invalid")
    return payload


def _type_prior_sha256(config: ResponseExperimentConfig) -> str:
    with Path(config.model.type_prior_path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _is_safe_checkpoint_value(value) -> bool:
    if value is None or isinstance(value, str | int | float | bool | torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str | int | float | bool)
            and _is_safe_checkpoint_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return all(_is_safe_checkpoint_value(item) for item in value)
    return False


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_REVISION",
    "MODEL_CONTRACT_REVISION",
    "ResponseCheckpointState",
    "ResponseCheckpointError",
    "inspect_response_checkpoint",
    "load_response_checkpoint",
    "save_response_checkpoint",
]
