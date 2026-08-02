from __future__ import annotations

from pathlib import Path

import torch
from pytest import MonkeyPatch
from torch import nn

import evaluation.response_pipeline as response_pipeline
import evaluation.rf_dynamic_conditioning as rf_dynamic_conditioning
import evaluation.rf_static as rf_static
from response_pipeline_history_fixtures import (
    fake_config,
    fake_model,
    fake_trainer,
    fit_glm,
    gradient_model,
    prepared_data,
    static_recorder,
)


def test_rf_finite_difference_toggle_controls_pipeline_extraction(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(response_pipeline, "fit_point_process_glm", fit_glm)
    seen: list[float | None] = []
    monkeypatch.setattr(response_pipeline, "extract_static_rf", static_recorder(seen))
    monkeypatch.setattr(
        rf_dynamic_conditioning,
        "extract_static_rf",
        static_recorder(seen),
    )

    (tmp_path / "default").mkdir()
    response_pipeline.evaluate_and_report_response_experiment(
        tmp_path / "default",
        model=fake_model(),
        initialized_model=fake_model(),
        trainer=fake_trainer(),
        data=prepared_data(),
        config=fake_config(),
        checkpoint=tmp_path / "checkpoint.pt",
    )
    assert seen
    assert set(seen) == {0.05}

    seen.clear()
    disabled = fake_config(rf_finite_difference_checks=False)
    (tmp_path / "disabled").mkdir()
    response_pipeline.evaluate_and_report_response_experiment(
        tmp_path / "disabled",
        model=fake_model(),
        initialized_model=fake_model(),
        trainer=fake_trainer(),
        data=prepared_data(),
        config=disabled,
        checkpoint=tmp_path / "checkpoint.pt",
    )

    assert seen
    assert set(seen) == {None}
    assert (tmp_path / "disabled" / "final_metrics.json").exists()
    assert (tmp_path / "disabled" / "rf_artifacts.pt").exists()
    assert (tmp_path / "disabled" / "run_manifest.json").exists()


def test_rf_finite_difference_toggle_skips_numerical_check(
    monkeypatch: MonkeyPatch,
) -> None:
    config = fake_config(rf_finite_difference_checks=False)

    def fail_check(
        model: nn.Module,
        sequence: torch.Tensor,
        kernel: torch.Tensor,
        epsilon: float,
        observed_counts: torch.Tensor | None,
    ) -> float:
        raise AssertionError("_finite_difference_check should not run")

    monkeypatch.setattr(rf_static, "_finite_difference_check", fail_check)

    result = rf_static.extract_static_rf(
        gradient_model(),
        torch.ones(1, 3, 1),
        lag_steps=2,
        finite_difference_tolerance=config.evaluation.finite_difference_tolerance,
    )

    assert result.finite_difference_relative_error == 0.0
    assert result.identifiable
