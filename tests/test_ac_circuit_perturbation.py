from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch
import pytest

from evaluation.mechanistic_retina.ac_circuit_perturbation import (
    run_ac_circuit_perturbation,
)
from evaluation.mechanistic_retina.ac_circuit_inputs import (
    ACCircuitInputError,
    rebuild_model,
)
from evaluation.mechanistic_retina.ac_circuit_support import (
    ACRFArtifactError,
    load_ac_rf_artifact,
)
from evaluation.mechanistic_retina.clean_sampled_benchmark import (
    CleanBenchmarkConfig,
    build_clean_benchmark,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
)


def _write_untrained_fixture(directory: Path) -> None:
    state = build_clean_benchmark(
        CleanBenchmarkConfig(
            train_stimuli=2,
            validation_stimuli=2,
            time_steps=20,
            trials=2,
            steps=1,
            checkpoint_steps=(0, 1),
            batch_size=2,
        )
    )
    directory.mkdir()
    model_config = asdict(state.student.config)
    model_config["architecture_mode"] = ArchitectureMode(
        state.student.config.architecture_mode
    ).value
    torch.save(
        {
            "role": "student-trained",
            "model_revision": MECHANISTIC_MODEL_REVISION,
            "model_state": state.student.state_dict(),
            "model_config": model_config,
            "config": asdict(state.config),
            "cone_positions": state.cone_positions,
            "cell_positions": state.cell_positions,
            "cell_types": state.cell_types,
            "polarities": state.polarities,
        },
        directory / "student-trained.pt",
    )
    torch.save(
        {
            "train_cones": state.train_cones,
            "validation_cones": state.validation_cones,
            "train_spikes": state.train_spikes,
            "validation_spikes": state.validation_spikes,
        },
        directory / "sampled-data.pt",
    )


def test_pre_learnable_temporal_checkpoint_is_rejected(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    _write_untrained_fixture(benchmark_dir)
    checkpoint = torch.load(
        benchmark_dir / "student-trained.pt",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["model_revision"] = MECHANISTIC_MODEL_REVISION - 1

    with pytest.raises(ACCircuitInputError, match="predates bounded-learnable"):
        rebuild_model(checkpoint)


def test_ac_perturbation_is_checkpoint_only_and_exactly_clamped(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "perturbation"
    _write_untrained_fixture(benchmark_dir)

    result = run_ac_circuit_perturbation(benchmark_dir, output_dir)
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    responses = torch.load(output_dir / "responses.pt", weights_only=True)
    rfs = load_ac_rf_artifact(output_dir / "rf-tensors.pt")

    assert result.parameter_state_unchanged
    assert payload["execution"]["training_performed"] is False
    assert payload["execution"]["optimizer_created"] is False
    assert payload["parameter_invariance"]["all_parameters_unchanged"] is True
    assert all(payload["parameter_invariance"][name] for name in ("H1", "BC", "RGC"))
    assert payload["validation"]["stimulus_count"] == 2
    assert payload["validation"]["trial_count"] == 2
    assert responses["normal"]["logits"].shape == (2, 2, 20, 8)
    assert responses["ac_structural_clamp"]["spike_probability"].shape == (
        2,
        2,
        20,
        8,
    )
    assert torch.count_nonzero(
        responses["ac_structural_clamp"]["ac_total_current"]
    ).item() == 0
    assert torch.equal(
        responses["ac_contribution"]["logits"],
        responses["normal"]["logits"]
        - responses["ac_structural_clamp"]["logits"],
    )
    assert torch.equal(
        rfs["normal"]["temporal"], rfs["normal"]["global"].sum(dim=-1)
    )
    assert torch.equal(
        rfs["normal"]["ac_pathway"],
        rfs["normal"]["global"] - rfs["ac_structural_clamp"]["global"],
    )
    assert torch.equal(
        rfs["ac_contribution"]["global"],
        rfs["normal"]["global"] - rfs["ac_structural_clamp"]["global"],
    )
    assert torch.equal(
        rfs["clamp_minus_normal"]["global"],
        rfs["ac_structural_clamp"]["global"] - rfs["normal"]["global"],
    )
    assert torch.count_nonzero(rfs["ac_structural_clamp"]["ac_pathway"]).item() == 0
    assert payload["response_change"]["mean_absolute_logit_change"] > 0
    assert payload["global_rf_change"]["difference_norm"] > 0
    assert payload["temporal_rf_change"]["difference_norm"] > 0
    identity = rfs["identity"]
    assert rfs["schema"] == "ac-circuit-perturbation-rf-v1"
    assert identity["lag_order_semantics"] == "oldest_to_current"
    assert identity["rf_estimand"].startswith("final-time conditional RGC-logit")
    assert identity["cell_order"] == list(range(8))
    assert identity["cone_order"] == list(range(state_cone_count := 16))
    assert len(identity["cone_positions_degs"]) == state_cone_count
    assert len(identity["cell_positions_degs"]) == 8
    assert identity["validation_split"] == "validation"
    assert set(identity["source_sha256"]) >= {
        "evaluation/mechanistic_retina/ac_circuit_perturbation.py",
        "evaluation/mechanistic_retina/rf_effective.py",
        "models/mechanistic_retina/pathway_gates.py",
    }
    assert payload["rf_artifact_identity"] == identity


def test_ac_rf_artifact_loader_rejects_lag_identity_drift(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "perturbation"
    _write_untrained_fixture(benchmark_dir)
    run_ac_circuit_perturbation(benchmark_dir, output_dir)
    artifact = torch.load(output_dir / "rf-tensors.pt", weights_only=True)
    artifact["identity"]["lag_order_semantics"] = "current_to_oldest"
    corrupt_path = tmp_path / "corrupt-rf.pt"
    torch.save(artifact, corrupt_path)

    with pytest.raises(ACRFArtifactError, match="lag order"):
        load_ac_rf_artifact(corrupt_path)


def test_ac_rf_artifact_loader_rejects_pathway_shape_drift(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "perturbation"
    _write_untrained_fixture(benchmark_dir)
    run_ac_circuit_perturbation(benchmark_dir, output_dir)
    artifact = torch.load(output_dir / "rf-tensors.pt", weights_only=True)
    artifact["ac_structural_clamp"]["ac_pathway"] = torch.tensor(0.0)
    corrupt_path = tmp_path / "corrupt-shape-rf.pt"
    torch.save(artifact, corrupt_path)

    with pytest.raises(ACRFArtifactError, match="invalid shapes"):
        load_ac_rf_artifact(corrupt_path)


def test_ac_perturbation_refuses_nonempty_output(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "perturbation"
    _write_untrained_fixture(benchmark_dir)
    output_dir.mkdir()
    (output_dir / "foreign.txt").write_text("keep", encoding="utf-8")

    try:
        run_ac_circuit_perturbation(benchmark_dir, output_dir)
    except FileExistsError as error:
        assert "output directory must be empty" in str(error)
    else:
        raise AssertionError("non-empty output directory was silently reused")
