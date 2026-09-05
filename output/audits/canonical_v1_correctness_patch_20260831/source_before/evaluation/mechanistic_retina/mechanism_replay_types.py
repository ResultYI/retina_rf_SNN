from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evaluation.mechanistic_retina.mechanism_checkpoints import SavedCheckpoint
from evaluation.mechanistic_retina.mechanism_replay_identity import ReplayContext
from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    MechanismRunEvidence,
    ProgressEvent,
)


@dataclass(frozen=True, slots=True)
class ReplayExecutionRequest:
    repo_root: Path
    evidence_dir: Path
    checkpoint_root: Path
    context: ReplayContext
    progress: Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class ReplayMetricComparison:
    ce_difference: float
    gate_difference: float
    pathway_rf_cosine_difference: float
    total_rf_global_difference: float
    exact_cell_fraction_difference: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CheckpointManifestEntry:
    saved: SavedCheckpoint
    relative_path: str
    roundtrip_state_equal: bool
    roundtrip_gate_difference: float


@dataclass(frozen=True, slots=True)
class ReplayRunSet:
    runs: tuple[MechanismRunEvidence, ...]
    checkpoints: tuple[CheckpointManifestEntry, ...]
    comparisons: tuple[ReplayMetricComparison, ...]

    def passed(self) -> bool:
        return all(value.passed for value in self.comparisons) and all(
            value.roundtrip_state_equal and value.roundtrip_gate_difference <= 1e-3
            for value in self.checkpoints
        )


@dataclass(frozen=True, slots=True)
class ReplayKey:
    teacher: str
    variant: AblationName
    seed: int


@dataclass(frozen=True, slots=True)
class ReplayRunError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


__all__ = [
    "CheckpointManifestEntry",
    "ReplayExecutionRequest",
    "ReplayKey",
    "ReplayMetricComparison",
    "ReplayRunError",
    "ReplayRunSet",
]
