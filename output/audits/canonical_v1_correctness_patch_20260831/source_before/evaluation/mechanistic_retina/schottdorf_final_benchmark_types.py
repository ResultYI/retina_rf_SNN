from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FinalBenchmarkConfig:
    repository_dir: Path
    movie_path: Path
    retinal_artifact_dir: Path
    glm_artifact_dir: Path
    output_dir: Path
    neural_maximum_steps: int = 2_000
    neural_patience: int = 200
    neural_learning_rate: float = 0.01
    neural_weight_decay: float = 1e-4
    cell_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FinalBenchmarkResult:
    artifact_dir: Path
    cell_count: int
    mean_validation_nll: dict[str, float]


class FinalBenchmarkError(RuntimeError):
    pass


__all__ = ["FinalBenchmarkConfig", "FinalBenchmarkError", "FinalBenchmarkResult"]
