from __future__ import annotations

import json
from pathlib import Path

import torch
from pytest import MonkeyPatch

import evaluation.response_pipeline as response_pipeline
from response_pipeline_history_fixtures import (
    dynamic_recorder,
    fake_config,
    fake_model,
    fake_trainer,
    fit_glm,
    free_static,
    history_recorder,
    prepared_data,
)


def test_pipeline_runs_conditional_and_free_running_rf_modes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    conditional_history_keys: list[str] = []
    monkeypatch.setattr(response_pipeline, "fit_point_process_glm", fit_glm)
    monkeypatch.setattr(
        response_pipeline,
        "conditional_rf_by_history",
        history_recorder(conditional_history_keys),
    )
    monkeypatch.setattr(
        response_pipeline,
        "evaluate_dynamic_rf",
        dynamic_recorder(),
    )
    monkeypatch.setattr(response_pipeline, "extract_static_rf", free_static())

    response_pipeline.evaluate_and_report_response_experiment(
        tmp_path,
        model=fake_model(),
        initialized_model=fake_model(),
        trainer=fake_trainer(),
        data=prepared_data(),
        config=fake_config(),
        checkpoint=tmp_path / "checkpoint.pt",
    )

    metrics = json.loads((tmp_path / "final_metrics.json").read_text())
    assert conditional_history_keys == ["zero", "matched_observed", "standard_train_rate"]
    assert metrics["evaluation_split"] == "validation"
    assert tuple(metrics["static_rf"]["by_history"]) == (
        "zero",
        "matched_observed",
        "standard_train_rate",
    )
    assert tuple(metrics["dynamic_rf"]["by_history"]) == (
        "zero",
        "matched_observed",
        "standard_train_rate",
    )
    assert metrics["static_rf"]["by_history"]["zero"]["trained"][
        "per_cell_kernel_norm"
    ] == [1.7320507764816284, 1.7320507764816284]
    assert metrics["dynamic_rf"]["by_history"]["matched_observed"]["trained"][
        "per_cell_signed_log_gain_shift"
    ] == [0.6931471824645996, 0.6931471824645996]
    artifacts = torch.load(
        tmp_path / "rf_artifacts.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(artifacts) == {
        "schema",
        "cell_ids",
        "cone_positions_degs",
        "lag_order",
        "conditional_static_by_history",
        "conditional_dynamic_by_history",
        "free_running",
    }
    assert artifacts["schema"] == "retina-rf-artifacts-v2"
    assert artifacts["cell_ids"] == ("cell",)
    assert tuple(artifacts["conditional_static_by_history"]) == (
        "zero",
        "matched_observed",
        "standard_train_rate",
    )
    assert artifacts["conditional_dynamic_by_history"]["zero"][
        "trained_low"
    ].shape == (2, 3, 1)
    assert artifacts["conditional_static_by_history"]["matched_observed"][
        "trained"
    ].shape == (2, 3, 1)
