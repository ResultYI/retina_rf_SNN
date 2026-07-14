from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import torch

from evaluation.checkpoint_metrics import (
    PopulationAblationMetrics,
    PopulationUsageMetrics,
    PredictionMetrics,
)
from evaluation.checkpoint_probes import RFProbeMetrics, TemporalProbeMetrics


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationConfig:
    checkpoint: Path
    normalization_stats: Path
    train_h5: tuple[Path, ...]
    eval_h5: tuple[Path, ...]
    output_dir: Path
    input_steps: int
    horizons: tuple[int, ...]
    batch_size: int
    device: torch.device
    rf_sample_count: int = 32
    glm_max_steps: int = 20
    humret_root: Path | None = None
    humret_model_grating: Path | None = None
    formal_evidence: bool = False

    def __post_init__(self) -> None:
        if min(self.input_steps, self.batch_size, self.rf_sample_count) < 1:
            raise CheckpointEvaluationError(
                "input_steps, batch_size, and rf_sample_count must be positive"
            )
        if self.glm_max_steps < 1 or not self.horizons:
            raise CheckpointEvaluationError(
                "glm_max_steps and at least one horizon are required"
            )
        if any(horizon < 1 for horizon in self.horizons):
            raise CheckpointEvaluationError("horizons must be positive")
        if not self.train_h5 or not self.eval_h5:
            raise CheckpointEvaluationError("train_h5 and eval_h5 are required")
        required = (
            self.checkpoint,
            self.normalization_stats,
            *self.train_h5,
            *self.eval_h5,
        )
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise CheckpointEvaluationError(f"Input file does not exist: {missing[0]}")
        if (self.humret_root is None) != (self.humret_model_grating is None):
            raise CheckpointEvaluationError(
                "humret_root and humret_model_grating must be provided together"
            )
        if self.humret_root is not None and not self.humret_root.is_dir():
            raise CheckpointEvaluationError("humret_root must be a directory")
        if (
            self.humret_model_grating is not None
            and not self.humret_model_grating.is_file()
        ):
            raise CheckpointEvaluationError("humret_model_grating must be a file")


class CheckpointMetadata(TypedDict):
    path: str
    stage: str
    epoch: int
    step: int


class EvaluationDataMetadata(TypedDict):
    normalization_stats: str
    train_h5: tuple[str, ...]
    eval_h5: tuple[str, ...]
    train_exports: int
    eval_exports: int
    eval_samples: int
    cone_count: int
    dt_ms: float
    input_steps: int
    horizons: tuple[int, ...]


class ParameterAuditPayload(TypedDict):
    name: str
    value: float
    lower: float
    upper: float
    boundary_fraction: float
    near_boundary: bool


class HumRetPayload(TypedDict):
    status: str
    reason: NotRequired[str]
    reference_root: NotRequired[str]
    model_grating_artifact: NotRequired[str]
    human_cells: NotRequired[int]
    model_units: NotRequired[int]
    mean_tuning_cosine_similarity: NotRequired[float]
    spatial_preference_total_variation: NotRequired[float]
    temporal_preference_total_variation: NotRequired[float]


class EvaluationArtifacts(TypedDict):
    summary: str
    rf_probes: str


class CheckpointEvaluationPayload(TypedDict):
    checkpoint: CheckpointMetadata
    data: EvaluationDataMetadata
    prediction: PredictionMetrics
    population_usage: tuple[PopulationUsageMetrics, ...]
    population_ablation: tuple[PopulationAblationMetrics, ...]
    temporal_probe_interpretation: str
    temporal_probes: tuple[TemporalProbeMetrics, ...]
    rf_probes: tuple[RFProbeMetrics, ...]
    parameter_audit: tuple[ParameterAuditPayload, ...]
    humret: HumRetPayload
    artifacts: EvaluationArtifacts
