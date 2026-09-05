from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from evaluation.mechanistic_retina.metrics import JsonValue


@dataclass(frozen=True, slots=True)
class SavedComparisonCheckpoint:
    path: Path
    sha256: str
    bytes: int


def save_comparison_checkpoint(
    path: Path, model: nn.Module, metadata: Mapping[str, JsonValue]
) -> SavedComparisonCheckpoint:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint write failed: {path}") from error
    return SavedComparisonCheckpoint(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
    )


def load_comparison_checkpoint(
    path: Path, model: nn.Module
) -> dict[str, JsonValue]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model_state_dict"]
        metadata = payload["metadata"]
        model.load_state_dict(state, strict=True)
        if not isinstance(metadata, dict):
            raise TypeError("checkpoint metadata must be a mapping")
        return dict(metadata)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"checkpoint load failed: {path}") from error


def write_json(path: Path, payload: JsonValue | Mapping[str, JsonValue]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, JsonValue]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_artifact_set(output: Path, required: Sequence[str]) -> None:
    actual = {path.name for path in output.iterdir() if path.is_file()}
    allowed = set(required) | {"failure.json"}
    if not set(required).issubset(actual) or not actual.issubset(allowed):
        raise ValueError("evidence file set differs from the canonical contract")
    for name in required:
        path = output / name
        if path.stat().st_size == 0:
            raise ValueError(f"empty evidence artifact: {name}")
        if name.endswith(".json") or name == "experiment-config.yaml":
            json.loads(path.read_text(encoding="utf-8"))
        elif name.endswith(".jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                json.loads(line)
        elif name.endswith(".csv"):
            with path.open(newline="", encoding="utf-8") as handle:
                if not csv.DictReader(handle).fieldnames:
                    raise ValueError(f"CSV has no header: {name}")


__all__ = [
    "SavedComparisonCheckpoint",
    "load_comparison_checkpoint",
    "save_comparison_checkpoint",
    "validate_artifact_set",
    "write_csv",
    "write_json",
]
