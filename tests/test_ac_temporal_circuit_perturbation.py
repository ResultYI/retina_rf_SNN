from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.ac_temporal_perturbation import (
    run_ac_temporal_circuit_perturbation,
)
from evaluation.mechanistic_retina.ac_temporal_probe import (
    TemporalProbeConfig,
    build_temporal_probe,
)
from evaluation.mechanistic_retina.ac_temporal_support import (
    ACTemporalArtifactError,
    validate_temporal_rf_artifact,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_metrics import (
    CellPerturbationRequest,
    cell_perturbation_metrics,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import (
    stimulus_onset_step,
)
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from tests.test_ac_circuit_perturbation import _write_untrained_fixture


def test_cell_perturbation_metrics_preserve_peak_and_rf_semantics() -> None:
    # Given two cells with known flash-evoked responses and proportional RFs.
    normal_logits = torch.tensor(
        [
            [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [1.0, -1.0]],
            [[0.0, 0.0], [2.0, -2.0], [1.0, -1.0], [0.0, 0.0]],
        ]
    )
    clamped_logits = 0.5 * normal_logits
    normal_rf = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    clamped_rf = 0.5 * normal_rf

    # When per-cell response and temporal-RF effects are computed.
    metrics = cell_perturbation_metrics(
        CellPerturbationRequest(
            normal_logits=normal_logits,
            clamped_logits=clamped_logits,
            normal_probability=torch.sigmoid(normal_logits),
            clamped_probability=torch.sigmoid(clamped_logits),
            normal_temporal_rf=normal_rf,
            clamped_temporal_rf=clamped_rf,
            baseline_steps=1,
            dt_ms=10.0,
        )
    )

    # Then magnitudes, forward-response latency, and RF lag metrics stay distinct.
    torch.testing.assert_close(
        metrics.mean_absolute_logit_change, torch.full((2,), 0.4375)
    )
    torch.testing.assert_close(
        metrics.normal_logit_peak_magnitude, torch.full((2,), 2.0)
    )
    torch.testing.assert_close(
        metrics.logit_peak_magnitude_change, torch.full((2,), -1.0)
    )
    torch.testing.assert_close(
        metrics.normal_logit_peak_latency_ms, torch.full((2,), 5.0)
    )
    torch.testing.assert_close(metrics.logit_peak_latency_change_ms, torch.zeros(2))
    torch.testing.assert_close(
        metrics.temporal_rf_normal_norm, torch.tensor((1.0, 2.0))
    )
    torch.testing.assert_close(
        metrics.temporal_rf_change_norm, torch.tensor((0.5, 1.0))
    )
    torch.testing.assert_close(metrics.temporal_rf_cosine, torch.ones(2))


def test_stimulus_onset_step_finds_first_nonzero_flash_frame() -> None:
    # Given a population batch with two pre-flash baseline frames.
    cones = torch.zeros(3, 6, 4)
    cones[:, 2:5] = 0.25

    # When the common flash onset is derived from the held-out stimulus tensor.
    onset = stimulus_onset_step(cones)

    # Then the first active frame is returned without NumPy-only tensor helpers.
    assert onset == 2


def test_temporal_probe_contains_distinct_chirp_and_flicker_drives() -> None:
    # Given
    positions = torch.tensor(
        [[-0.05, -0.05], [0.05, -0.05], [-0.05, 0.05], [0.05, 0.05]]
    )
    config = TemporalProbeConfig(time_steps=64, baseline_steps=8)

    # When
    probe = build_temporal_probe(positions, 5.0, config)

    # Then
    assert probe.names == ("linear_chirp", "multifrequency_flicker")
    assert probe.cone_response.shape == (2, 64, 4)
    assert probe.temporal_drive.shape == (2, 64)
    assert torch.count_nonzero(probe.cone_response[:, :8]).item() == 0
    assert torch.count_nonzero(probe.cone_response[:, 8:]).item() > 0
    assert not torch.equal(probe.temporal_drive[0], probe.temporal_drive[1])
    assert torch.isfinite(probe.cone_response).all()


def test_ac_temporal_perturbation_saves_checkpoint_only_circuit_effect(
    tmp_path: Path,
) -> None:
    # Given
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "temporal-perturbation"
    _write_untrained_fixture(benchmark_dir)
    config = TemporalProbeConfig(time_steps=64, baseline_steps=8)

    # When
    result = run_ac_temporal_circuit_perturbation(
        benchmark_dir,
        output_dir,
        probe_config=config,
    )

    # Then
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    stimuli = torch.load(output_dir / "stimuli.pt", weights_only=True)
    responses = torch.load(output_dir / "responses.pt", weights_only=True)
    rfs = torch.load(output_dir / "rf-tensors.pt", weights_only=True)
    assert result.parameter_state_unchanged
    assert payload["execution"] == {
        "training_performed": False,
        "optimizer_created": False,
        "checkpoint_role": "student-trained",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "clamps": ["no-amacrine-local", "no-amacrine-transient"],
    }
    assert payload["probe"]["names"] == [
        "linear_chirp",
        "multifrequency_flicker",
    ]
    assert payload["probe"]["observed_history_context"] == "all-zero"
    assert stimuli["cone_response"].shape == (2, 64, 16)
    assert responses["normal"]["logits"].shape == (2, 1, 64, 8)
    assert responses["ac_structural_clamp"]["spike_probability"].shape == (
        2,
        1,
        64,
        8,
    )
    assert torch.equal(
        responses["ac_contribution"]["logits"],
        responses["normal"]["logits"] - responses["ac_structural_clamp"]["logits"],
    )
    assert (
        torch.count_nonzero(responses["ac_structural_clamp"]["ac_total_current"]).item()
        == 0
    )
    assert rfs["normal"]["temporal"].shape == (2, 8, 16)
    assert rfs["identity"]["model_revision"] == MECHANISTIC_MODEL_REVISION
    assert rfs["identity"]["temporal_rf_definition"] == (
        "signed sum of final-bin global RF over cone dimension"
    )
    assert torch.equal(
        rfs["ac_contribution"]["temporal"],
        rfs["normal"]["temporal"] - rfs["ac_structural_clamp"]["temporal"],
    )
    assert torch.count_nonzero(rfs["ac_structural_clamp"]["ac_pathway"]).item() == 0
    invariance = payload["parameter_invariance"]
    assert invariance["all_parameters_unchanged"] is True
    assert invariance["all_state_tensors_unchanged"] is True
    assert all(invariance[name] for name in ("weights", "gates", "tau", "delay"))
    assert payload["structural_clamp"]["ac_current_exact_zero"] is True
    assert payload["timing_contract"]["rf_lag_window"] == {
        "lag_steps": 16,
        "dt_ms": 5.0,
        "learnable": False,
    }
    assert payload["timing_contract"]["rgc_history_shift"] == {
        "shift_steps": 1,
        "shift_ms": 5.0,
        "learnable": False,
    }
    assert payload["response_change"]["mean_absolute_logit_change"] > 0
    assert payload["response_change"]["mean_absolute_probability_change"] > 0
    assert payload["peak_latency_change"]["logit_peak_absolute_change"] != 0
    assert payload["temporal_rf_change"]["difference_norm"] > 0
    assert set(payload["lineage"]["source_sha256"]) >= {
        "evaluation/mechanistic_retina/ac_temporal_perturbation.py",
        "evaluation/mechanistic_retina/ac_temporal_probe.py",
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "results.json",
        "stimuli.pt",
        "responses.pt",
        "rf-tensors.pt",
    }


def test_temporal_rf_validator_rejects_pathway_decomposition_drift(
    tmp_path: Path,
) -> None:
    # Given
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "temporal-perturbation"
    _write_untrained_fixture(benchmark_dir)
    run_ac_temporal_circuit_perturbation(
        benchmark_dir,
        output_dir,
        probe_config=TemporalProbeConfig(time_steps=32, baseline_steps=4),
    )
    artifact = torch.load(output_dir / "rf-tensors.pt", weights_only=True)
    artifact["ac_contribution"]["ac_pathway"] = torch.zeros_like(
        artifact["ac_contribution"]["ac_pathway"]
    )

    # When/Then
    with pytest.raises(ACTemporalArtifactError, match="decomposition"):
        validate_temporal_rf_artifact(artifact)


def test_temporal_rf_validator_rejects_schema_revision_drift(
    tmp_path: Path,
) -> None:
    # Given
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "temporal-perturbation"
    _write_untrained_fixture(benchmark_dir)
    run_ac_temporal_circuit_perturbation(
        benchmark_dir,
        output_dir,
        probe_config=TemporalProbeConfig(time_steps=32, baseline_steps=4),
    )
    artifact = torch.load(output_dir / "rf-tensors.pt", weights_only=True)
    artifact["schema_revision"] = 2

    # When/Then
    with pytest.raises(ACTemporalArtifactError, match="schema"):
        validate_temporal_rf_artifact(artifact)
