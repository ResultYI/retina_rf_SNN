from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.mechanism_run_types import AblationName
from evaluation.mechanistic_retina.metrics import JsonValue
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    architecture_revision: str
    teacher_id: str
    teacher_hash: str
    condition: str
    structural_variant: AblationName
    seed: int
    step: int
    run_id: str
    dataset_identity: str
    cell_order: tuple[str, ...]
    cone_order: tuple[int, ...]
    lag_order: tuple[int, ...]
    gate_values: Mapping[str, float]
    config_snapshot: Mapping[str, JsonValue]
    config_hash: str
    source_hash: str


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    path: Path
    sha256: str
    bytes: int
    identity: CheckpointIdentity


@dataclass(frozen=True, slots=True)
class CheckpointError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def save_final_checkpoint(
    path: Path,
    model: MechanisticGraphTemporalRetina,
    identity: CheckpointIdentity,
) -> SavedCheckpoint:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(_metadata_payload(identity)),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        raise CheckpointError(f"checkpoint write failed: {path}") from error
    return SavedCheckpoint(path, sha256_file(path), path.stat().st_size, identity)


def load_final_checkpoint(
    path: Path,
    model: MechanisticGraphTemporalRetina,
) -> CheckpointIdentity:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return _parse_identity(payload["metadata"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CheckpointError(f"checkpoint load failed: {path}") from error


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensors_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _metadata_payload(identity: CheckpointIdentity) -> Mapping[str, JsonValue]:
    return {
        "architecture_revision": identity.architecture_revision,
        "teacher_id": identity.teacher_id,
        "teacher_hash": identity.teacher_hash,
        "condition": identity.condition,
        "structural_variant": identity.structural_variant.value,
        "seed": identity.seed,
        "step": identity.step,
        "run_id": identity.run_id,
        "dataset_identity": identity.dataset_identity,
        "cell_order": list(identity.cell_order),
        "cone_order": list(identity.cone_order),
        "lag_order": list(identity.lag_order),
        "gate_values": dict(identity.gate_values),
        "config_snapshot": dict(identity.config_snapshot),
        "config_hash": identity.config_hash,
        "source_hash": identity.source_hash,
    }


def _parse_identity(payload) -> CheckpointIdentity:
    return CheckpointIdentity(
        architecture_revision=str(payload["architecture_revision"]),
        teacher_id=str(payload["teacher_id"]),
        teacher_hash=str(payload["teacher_hash"]),
        condition=str(payload["condition"]),
        structural_variant=AblationName(str(payload["structural_variant"])),
        seed=int(payload["seed"]),
        step=int(payload["step"]),
        run_id=str(payload["run_id"]),
        dataset_identity=str(payload["dataset_identity"]),
        cell_order=tuple(str(value) for value in payload["cell_order"]),
        cone_order=tuple(int(value) for value in payload["cone_order"]),
        lag_order=tuple(int(value) for value in payload["lag_order"]),
        gate_values={str(key): float(value) for key, value in payload["gate_values"].items()},
        config_snapshot=dict(payload["config_snapshot"]),
        config_hash=str(payload["config_hash"]),
        source_hash=str(payload["source_hash"]),
    )


__all__ = [
    "CheckpointError",
    "CheckpointIdentity",
    "SavedCheckpoint",
    "load_final_checkpoint",
    "save_final_checkpoint",
    "sha256_file",
    "tensors_sha256",
]
