from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Phase1RunConfig:
    architecture_id: str
    candidate0_path: Path
    checkpoint_path: Path
    seeds: tuple[int, ...]
    smoke_seed: int
    smoke_steps: int
    smoke_checkpoints: tuple[int, ...]
    steps: int
    checkpoints: tuple[int, ...]
    learning_rate: float
    batch_size: int
    neural_operators_enabled: bool
    dynamic_modulation_enabled: bool
    monitor_interval_seconds: float
    poll_interval_seconds: float


def load_phase1_run_config(path: Path) -> Phase1RunConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Phase1RunConfig(
        str(payload["architecture_id"]),
        Path(payload["candidate0_path"]),
        Path(payload["checkpoint_path"]),
        tuple(int(value) for value in payload["seeds"]),
        int(payload["smoke_seed"]),
        int(payload["smoke_steps"]),
        tuple(int(value) for value in payload["smoke_checkpoints"]),
        int(payload["steps"]),
        tuple(int(value) for value in payload["checkpoints"]),
        float(payload["learning_rate"]),
        int(payload["batch_size"]),
        bool(payload["neural_operators_enabled"]),
        bool(payload["dynamic_modulation_enabled"]),
        float(payload["monitor_interval_seconds"]),
        float(payload["poll_interval_seconds"]),
    )


__all__ = ["Phase1RunConfig", "load_phase1_run_config"]
