from __future__ import annotations

import numpy as np
import torch
from torch import nn

from baselines.point_process_glm import GLMFitResult, PointProcessGLM
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from evaluation.response_metrics import ResponseMetrics
from evaluation.response_report_schema import RFModeEvidence
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_static import StaticRFResult
import evaluation.response_pipeline as response_pipeline
from training.response_config import (
    ResponseDataConfig,
    ResponseEvaluationConfig,
    ResponseExperimentConfig,
    ResponseModelConfig,
    ResponseTrainingConfig,
)
from training.response_data import PreparedResponseData, ResponseSplit
from training.response_trainer import ResponseHistoryMode


def dynamic_recorder():
    def fake_dynamic(
        model: nn.Module,
        split: ResponseSplit,
        *,
        lag_steps: int,
        condition_on_observed: bool = True,
        history_mode: str | None = None,
        standard_history_counts: torch.Tensor | None = None,
        recovery_delays_ms: tuple[int, ...] = (0,),
        dt_ms: float = 5.0,
        bootstrap_iterations: int = 1000,
        seed: int = 0,
        teacher_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
        teacher_context_gain_envelope: torch.Tensor | None = None,
        finite_difference_tolerance: float | None = 0.05,
    ) -> DynamicRFResult:
        return dynamic_rf()

    return fake_dynamic


def history_recorder(keys: list[str]):
    def fake_history(
        model: nn.Module,
        initialized_model: nn.Module,
        split: ResponseSplit,
        config: ResponseExperimentConfig,
        dt_ms: float,
        seed: int,
        teacher_dynamic: tuple[torch.Tensor, torch.Tensor] | None,
        teacher_envelope: torch.Tensor | None,
        standard_history_counts: torch.Tensor | None = None,
    ) -> dict[str, RFModeEvidence]:
        keys.extend(("zero", "matched_observed", "standard_train_rate"))
        return {
            key: RFModeEvidence(
                static_rf(),
                static_rf(),
                None,
                dynamic_rf(),
                dynamic_rf(),
                comparison(),
            )
            for key in keys
        }

    return fake_history


def free_static():
    def fake_static(
        model: nn.Module,
        sequence: torch.Tensor,
        *,
        lag_steps: int,
        finite_difference_tolerance: float | None = 0.05,
    ) -> StaticRFResult:
        return static_rf()

    return fake_static


def static_recorder(seen: list[float | None]):
    missing = -1.0

    def fake_static(
        model: nn.Module,
        sequence: torch.Tensor,
        *,
        lag_steps: int,
        observed_counts: torch.Tensor | None = None,
        epsilon: float = 1e-3,
        finite_difference_tolerance: float | None = missing,
    ) -> StaticRFResult:
        seen.append(finite_difference_tolerance)
        return static_rf()

    return fake_static


def fit_glm(
    data: PreparedResponseData,
    *,
    device: torch.device,
    burn_in_steps: int = 0,
    evaluate_test: bool = False,
) -> GLMFitResult:
    metrics = response_metrics(0.6)
    return GLMFitResult(
        PointProcessGLM(1, 1, 1),
        metrics,
        metrics if evaluate_test else None,
        1,
    )


def response_metrics(nll: float) -> ResponseMetrics:
    return ResponseMetrics(nll, 0.1, 0.2, 0.3, 0.4, (nll,))


def static_rf() -> StaticRFResult:
    return StaticRFResult(torch.ones(2, 3, 1), 0.01, True)


def dynamic_rf() -> DynamicRFResult:
    return DynamicRFResult(
        pair_count=3,
        mean_shape_distance=0.2,
        mean_log_gain_shift=0.2,
        shape_distance_ci=(0.1, 0.3),
        gain_shift_ci=(0.1, 0.3),
        reset_shape_distance=0.0,
        recovery_shape_distances=(0.2, 0.1),
        finite_difference_relative_error=0.01,
        teacher_shape_error=None,
        teacher_gain_error=None,
        per_source_shape_distances=(0.2, 0.2, 0.2),
        per_source_gain_shifts=(0.2, 0.2, 0.2),
        status="not_supported",
        mean_low_kernel=torch.ones(2, 3, 1),
        mean_high_kernel=torch.full((2, 3, 1), 2.0),
    )


def comparison():
    return response_pipeline.compare_dynamic_rf(dynamic_rf(), dynamic_rf())


def fake_model() -> nn.Module:
    return _FakeModel()


def gradient_model() -> nn.Module:
    return _GradientModel()


def fake_trainer():
    return _FakeTrainer()


def prepared_data() -> PreparedResponseData:
    cells = CellMetadata(
        ids=("cell",),
        type_ids=("midget",),
        polarities=np.zeros(1, dtype=np.int64),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        eccentricities_deg=np.ones(1, dtype=np.float32),
    )
    validation = ResponseSplit(
        cone_response=torch.ones(6, 4, 1),
        spike_counts=torch.zeros(6, 2, 4, 1),
        valid_mask=torch.ones(6, 2, 4, 1, dtype=torch.bool),
        source_ids=("a", "a", "b", "b", "c", "c"),
        context_ids=("low", "high", "low", "high", "low", "high"),
    )
    return PreparedResponseData(
        train=validation,
        validation=validation,
        test=validation,
        cells=cells,
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        time_axis_seconds=np.arange(4, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(1, dtype=np.float32),
        normalization_std=np.ones(1, dtype=np.float32),
        fingerprint="fingerprint",
        input_identity=synthetic_input_identity(1, ("source",)),
    )


def fake_config(*, rf_finite_difference_checks: bool = True) -> ResponseExperimentConfig:
    return ResponseExperimentConfig(
        seed=7,
        data=ResponseDataConfig("none", "none", "none", 4),
        model=ResponseModelConfig("priors.yaml", 0.1, 20.0, 5.0),
        training=ResponseTrainingConfig(1, 3, 1, 1, 1, 0.001, 1.0, 1),
        evaluation=ResponseEvaluationConfig(
            2,
            (0, 5),
            rf_finite_difference_checks=rf_finite_difference_checks,
        ),
    )


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgc = _FakeRGC()
        self.weight = nn.Parameter(torch.tensor(1.0))


class _FakeRGC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_mask", torch.ones(1, 1, dtype=torch.bool))


class _GradientOutput:
    def __init__(self, spike_logits: torch.Tensor) -> None:
        self.spike_logits = spike_logits


class _GradientModel(_FakeModel):
    def forward_sequence(
        self,
        stimulus: torch.Tensor,
        *,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[_GradientOutput, None]:
        return _GradientOutput(stimulus), None


class _FakeTrainer:
    device = torch.device("cpu")

    def evaluate(
        self,
        split: ResponseSplit,
        *,
        history_mode: ResponseHistoryMode = "observed",
        model: nn.Module | None = None,
    ) -> ResponseMetrics:
        return response_metrics(
            {
                "observed": 0.4,
                "zero": 0.3,
                "shuffled": 0.45,
                "free_running": 0.5,
            }[history_mode]
        )
