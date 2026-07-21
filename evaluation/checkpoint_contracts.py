from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import torch

from evaluation.checkpoint_metrics import (
    PopulationAblationMetrics,
    PopulationUsageMetrics,
    ReconstructionMetrics,
)
from evaluation.checkpoint_probes import RFProbeMetrics
from evaluation.temporal_probes import TemporalProbeMetrics


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
    batch_size: int
    device: torch.device
    rf_sample_count: int = 32
    glm_max_steps: int = 20
    humret_root: Path | None = None
    humret_model_response: Path | None = None
    formal_evidence: bool = False

    def __post_init__(self) -> None:
        if min(self.input_steps, self.batch_size, self.rf_sample_count) < 1:
            raise CheckpointEvaluationError(
                "input_steps, batch_size, and rf_sample_count must be positive"
            )
        if self.glm_max_steps < 1:
            raise CheckpointEvaluationError("glm_max_steps must be positive")
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
        if (self.humret_root is None) != (self.humret_model_response is None):
            raise CheckpointEvaluationError(
                "humret_root and humret_model_response must be provided together"
            )
        if self.humret_root is not None and not self.humret_root.is_dir():
            raise CheckpointEvaluationError("humret_root must be a directory")
        if (
            self.humret_model_response is not None
            and not self.humret_model_response.is_file()
        ):
            raise CheckpointEvaluationError("humret_model_response must be a file")
        if (
            self.humret_model_response is not None
            and self.humret_model_response.suffix.lower() != ".npz"
        ):
            raise CheckpointEvaluationError("humret_model_response must be an .npz file")


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


class ParameterAuditPayload(TypedDict):
    name: str
    value: float
    lower: float
    upper: float
    boundary_fraction: float
    near_boundary: bool


class ContextAuditPayload(TypedDict):
    input_steps: int
    dt_ms: float
    tau_upper_ms: float
    initialization_residual_bound: float
    analytic_sufficient: bool
    empirical_status: Literal[
        "passed",
        "failed",
        "not_run_insufficient_history",
    ]
    empirical_sufficient: bool
    empirical_reason: NotRequired[str]
    midget_rate_relative_rms: NotRequired[float]
    parasol_rate_relative_rms: NotRequired[float]
    current_reconstruction_relative_rms: NotRequired[float]


class ArchitectureCompliancePayload(TypedDict):
    midget_sustained_fraction: float
    midget_transient_fraction: float
    parasol_sustained_fraction: float
    parasol_transient_fraction: float
    kinetic_order_ok: bool


class RFProbeStatusPayload(TypedDict):
    status: Literal["run", "skipped", "not_identifiable"]
    reason: str


HumRetMetricPayload = TypedDict(
    "HumRetMetricPayload",
    {
        "model_human_distance": float,
        "human_split_half_p95": float,
        "pass": bool,
    },
)


class HumRetPayload(TypedDict):
    status: Literal["not_run", "ok"]
    reason: NotRequired[str]
    reference_root: NotRequired[str]
    model_response_artifact: NotRequired[str]
    checkpoint_sha256: NotRequired[str]
    frontend: NotRequired[Literal["ISETBio"]]
    protocol_version: NotRequired[Literal["humret_functional_v1"]]
    human_grating_cells: NotRequired[int]
    human_chirp_cells: NotRequired[int]
    model_units: NotRequired[int]
    grating_mean_tuning_cosine_distance: NotRequired[HumRetMetricPayload]
    grating_spatial_preference_total_variation: NotRequired[HumRetMetricPayload]
    grating_temporal_preference_total_variation: NotRequired[HumRetMetricPayload]
    chirp_mean_waveform_cosine_distance: NotRequired[HumRetMetricPayload]
    chirp_peak_frequency_total_variation: NotRequired[HumRetMetricPayload]
    external_functional_pass: NotRequired[bool]
    bootstrap_seed: NotRequired[int]
    bootstrap_iterations: NotRequired[int]
    interpretation: NotRequired[
        Literal["functional_population_distribution_only"]
    ]


class EvaluationArtifacts(TypedDict):
    summary: str
    rf_probes: NotRequired[str]


class CheckpointEvaluationPayload(TypedDict):
    evidence_class: Literal["formal_candidate", "non_formal_smoke"]
    context_audit: ContextAuditPayload
    architecture_compliance: ArchitectureCompliancePayload
    rf_probe_status: RFProbeStatusPayload
    checkpoint: CheckpointMetadata
    data: EvaluationDataMetadata
    reconstruction: ReconstructionMetrics
    population_usage: tuple[PopulationUsageMetrics, ...]
    population_ablation: tuple[PopulationAblationMetrics, ...]
    temporal_probe_interpretation: str
    temporal_probes: tuple[TemporalProbeMetrics, ...]
    rf_probes: tuple[RFProbeMetrics, ...]
    parameter_audit: tuple[ParameterAuditPayload, ...]
    humret: HumRetPayload
    artifacts: EvaluationArtifacts
