from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from pytest import MonkeyPatch
from torch import nn

from baselines.point_process_glm import GLMFitResult, PointProcessGLM
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from data.synthetic_teacher import load_teacher_rf_metadata
import evaluation.response_pipeline as response_pipeline
from evaluation.response_metrics import ResponseMetrics
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_static import StaticRFResult
from training.response_config import (
    ResponseDataConfig,
    ResponseEvaluationConfig,
    ResponseExperimentConfig,
    ResponseModelConfig,
    ResponseTrainingConfig,
)
from training.response_data import PreparedResponseData, ResponseSplit
from training.response_trainer import ResponseHistoryMode


def test_pipeline_runs_conditional_and_free_running_rf_modes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    dynamic_modes: list[bool] = []
    dynamic_splits: list[str] = []
    static_modes: list[bool] = []
    monkeypatch.setattr(response_pipeline, "fit_point_process_glm", _fit_glm)
    monkeypatch.setattr(
        response_pipeline,
        "evaluate_dynamic_rf",
        _dynamic_recorder(dynamic_modes, dynamic_splits),
    )
    monkeypatch.setattr(response_pipeline, "trial_conditioned_rf", _conditional_static(static_modes))
    monkeypatch.setattr(response_pipeline, "extract_static_rf", _free_static(static_modes))

    # When
    response_pipeline.evaluate_and_report_response_experiment(
        tmp_path,
        model=_FakeModel(),
        initialized_model=_FakeModel(),
        trainer=_FakeTrainer(),
        data=_prepared_data(),
        config=_config(),
        checkpoint=tmp_path / "checkpoint.pt",
    )

    # Then
    metrics = json.loads((tmp_path / "final_metrics.json").read_text())
    assert dynamic_modes == [True, True, False, False]
    assert dynamic_splits == ["validation"] * 4
    assert static_modes == [True, True, False, False]
    assert metrics["evaluation_split"] == "validation"
    assert metrics["response_prediction"]["glm_test"] is None
    assert metrics["response_prediction"]["initialized_conditional"]["nll"] == 0.4
    history = metrics["response_prediction"]["history_diagnostic"]
    assert history["zero"]["nll"] == 0.3
    assert history["shuffled"]["nll"] == 0.45
    assert metrics["parameter_delta_audit"][0]["name"] == "weight"
    parameter_delta = json.loads((tmp_path / "parameter_delta.json").read_text())
    assert parameter_delta == metrics["parameter_delta_audit"]
    assert "conditional" in metrics["dynamic_rf"]
    assert "free_running" in metrics["dynamic_rf"]
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["evaluation_split"] == "validation"
    artifacts = torch.load(
        tmp_path / "rf_artifacts.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert artifacts["cell_ids"] == ("cell",)
    assert artifacts["conditional_dynamic_trained_low"].shape == (2, 3, 1)
    assert artifacts["conditional_static_trained"].shape == (2, 3, 1)


def test_synthetic_rf_metadata_loader_reads_kernels_and_recovery(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as handle:
        teacher = handle.create_group("teacher")
        teacher.create_dataset("static_kernel", data=np.ones((2, 3, 1), dtype=np.float32))
        teacher.create_dataset("context_kernel_low", data=np.ones((2, 3, 1), dtype=np.float32))
        teacher.create_dataset(
            "context_kernel_high",
            data=np.ones((2, 3, 1), dtype=np.float32) * 2,
        )
        teacher.create_dataset(
            "context_gain_envelope",
            data=np.ones((4, 5, 2), dtype=np.float32),
        )

    # When
    metadata = load_teacher_rf_metadata(path)

    # Then
    assert metadata is not None
    assert metadata.static_kernel.shape == (2, 3, 1)
    assert metadata.context_gain_envelope is not None
    assert metadata.context_gain_envelope.shape == (4, 5, 2)


def _dynamic_recorder(modes: list[bool], splits: list[str]):
    def fake_dynamic(
        model: nn.Module,
        split: ResponseSplit,
        *,
        lag_steps: int,
        condition_on_observed: bool = True,
        recovery_delays_ms: tuple[int, ...] = (0,),
        dt_ms: float = 5.0,
        bootstrap_iterations: int = 1000,
        seed: int = 0,
        teacher_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
        teacher_context_gain_envelope: torch.Tensor | None = None,
    ) -> DynamicRFResult:
        modes.append(condition_on_observed)
        splits.append(split.source_ids[0])
        return _dynamic()

    return fake_dynamic


def _conditional_static(modes: list[bool]):
    def fake_static(
        model: nn.Module,
        split: ResponseSplit,
        index: int,
        lag_steps: int,
    ) -> StaticRFResult:
        modes.append(True)
        return _static_rf()

    return fake_static


def _free_static(modes: list[bool]):
    def fake_static(
        model: nn.Module,
        sequence: torch.Tensor,
        *,
        lag_steps: int,
    ) -> StaticRFResult:
        modes.append(False)
        return _static_rf()

    return fake_static


def _metrics(nll: float) -> ResponseMetrics:
    return ResponseMetrics(nll, 0.1, 0.2, 0.3, 0.4, (nll,))


def _fit_glm(
    data: PreparedResponseData,
    *,
    device: torch.device,
    burn_in_steps: int = 0,
    evaluate_test: bool = False,
) -> GLMFitResult:
    metrics = _metrics(0.6)
    return GLMFitResult(
        PointProcessGLM(1, 1, 1),
        metrics,
        metrics if evaluate_test else None,
        1,
    )


def _static_rf() -> StaticRFResult:
    return StaticRFResult(torch.ones(2, 3, 1), 0.01, True)


def _dynamic() -> DynamicRFResult:
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


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgc = _FakeRGC()
        self.weight = nn.Parameter(torch.tensor(1.0))


class _FakeRGC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_mask", torch.ones(1, 1, dtype=torch.bool))


class _FakeTrainer:
    device = torch.device("cpu")

    def evaluate(
        self,
        split: ResponseSplit,
        *,
        history_mode: ResponseHistoryMode = "observed",
        model: _FakeModel | None = None,
    ) -> ResponseMetrics:
        return _metrics(
            {
                "observed": 0.4,
                "zero": 0.3,
                "shuffled": 0.45,
                "free_running": 0.5,
            }[history_mode]
        )


def _prepared_data() -> PreparedResponseData:
    cells = CellMetadata(
        ids=("cell",),
        type_ids=("midget",),
        polarities=np.zeros(1, dtype=np.int64),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        eccentricities_deg=np.ones(1, dtype=np.float32),
    )
    validation = ResponseSplit(
        cone_response=torch.ones(1, 4, 1),
        spike_counts=torch.zeros(1, 2, 4, 1),
        valid_mask=torch.ones(1, 2, 4, 1, dtype=torch.bool),
        source_ids=("validation",),
        context_ids=("stationary",),
    )
    test = ResponseSplit(
        cone_response=validation.cone_response,
        spike_counts=validation.spike_counts,
        valid_mask=validation.valid_mask,
        source_ids=("test",),
        context_ids=validation.context_ids,
    )
    return PreparedResponseData(
        train=validation,
        validation=validation,
        test=test,
        cells=cells,
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        time_axis_seconds=np.arange(4, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(1, dtype=np.float32),
        normalization_std=np.ones(1, dtype=np.float32),
        fingerprint="fingerprint",
        input_identity=synthetic_input_identity(1, ("source",)),
    )


def _config() -> ResponseExperimentConfig:
    return ResponseExperimentConfig(
        seed=7,
        data=ResponseDataConfig("none", "none", "none", 4),
        model=ResponseModelConfig("priors.yaml", 0.1, 20.0, 5.0),
        training=ResponseTrainingConfig(1, 3, 1, 1, 1, 0.001, 1.0, 1),
        evaluation=ResponseEvaluationConfig(2, (0, 5)),
    )
