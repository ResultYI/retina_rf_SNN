from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class SchottdorfBaselineRunConfig:
    repository_dir: Path
    movie_path: Path
    retinal_artifact_dir: Path
    output_dir: Path
    cell_ids: tuple[str, ...] | None = None
    glm_max_iterations: int = 2_000


@dataclass(frozen=True, slots=True)
class SchottdorfBaselineRunResult:
    artifact_dir: Path
    cell_count: int
    constant_rate_nll: float
    glm_nll: float
    retinal_nll: float


@dataclass(frozen=True, slots=True)
class SchottdorfBaselineRunError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


class ParameterCounts(TypedDict):
    total: int
    requires_grad: int
    optimizer_listed: int


class BaselineCellRecord(TypedDict):
    cell_id: str
    recording_ids: list[str]
    recording_count: int
    retinal_class: str
    canonical_cell_type: str
    polarity: str
    native_dt_ms: float
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    time_segment_disjoint: bool
    constant_rate_nll: float
    glm_nll: float
    retinal_nll: float
    winner: str
    retinal_strictly_better_than_constant: bool
    glm_train_nll_initial: float
    glm_train_nll_trained: float
    glm_solver_iterations: int
    glm_solver_evaluations: int
    glm_final_gradient_max: float
    glm_strict_gradient_converged: bool
    glm_solver_terminated_before_budget: bool
    glm_converged: bool
    glm_gradients_finite: bool
    glm_actually_updated: list[str]
    constant_parameters: ParameterCounts
    glm_parameters: ParameterCounts
    retinal_parameters: ParameterCounts
    retinal_nll_replay_error: float
    source_checkpoint: str
    source_checkpoint_sha256_before: str
    source_checkpoint_sha256_after: str


__all__ = [
    "BaselineCellRecord",
    "ParameterCounts",
    "SchottdorfBaselineRunConfig",
    "SchottdorfBaselineRunError",
    "SchottdorfBaselineRunResult",
]
