from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.mechanistic_retina.karamanlis_v1_ac_artifacts import (
    PerturbationArtifactRequest,
    TemporalRFIdentityRequest,
    build_temporal_rf_identity,
    save_perturbation_artifacts,
    validate_v1_ac_temporal_rf_artifact,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import (
    CollectedResponses,
    ResponseTensors,
    V1ACRuntimeError,
    validate_ac_clamp,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)


def _responses(clamped_local: torch.Tensor, clamped_transient: torch.Tensor):
    logits = torch.tensor([[[0.0], [1.0]]])
    normal = ResponseTensors(
        logits,
        torch.sigmoid(logits),
        torch.ones_like(logits),
        torch.ones_like(logits),
    )
    clamped = ResponseTensors(
        logits * 0.5,
        torch.sigmoid(logits * 0.5),
        clamped_local,
        clamped_transient,
    )
    return CollectedResponses(normal, clamped, True)


def test_ac_clamp_contract_rejects_equal_and_opposite_currents() -> None:
    # Given two nonzero clamped AC currents whose sum cancels exactly.
    local = torch.ones(1, 2, 1)
    transient = -local

    # When the structural-clamp contract is validated.
    # Then each pathway must be exact-zero independently.
    with pytest.raises(V1ACRuntimeError, match="structural clamp"):
        validate_ac_clamp(_responses(local, transient), state_unchanged=True)


def test_v1_ac_temporal_rf_artifact_carries_full_identity(tmp_path: Path) -> None:
    # Given a valid simultaneous AC clamp and an explicit real-data RF context.
    zeros = torch.zeros(1, 2, 1)
    responses = _responses(zeros, zeros)
    identity = build_temporal_rf_identity(
        TemporalRFIdentityRequest(
            checkpoint_sha256="checkpoint-hash",
            checkpoint_stage="best_trained",
            checkpoint_best_step=260,
            training_seed=20260302,
            model_revision=3,
            dt_ms=11.764705882352942,
            lag_ms=torch.tensor((0.0, 11.764705882352942)),
            cell_ids=("cell-1",),
            cell_types=("midget",),
            polarities=("ON",),
            context_indices=(0,),
            source_image_ids=("image-1",),
            trial_indices=(7,),
            source_sha256={"analysis.py": "source-hash"},
        )
    )

    # When the response and temporal-RF artifacts are saved.
    save_perturbation_artifacts(
        PerturbationArtifactRequest(
            tmp_path,
            responses,
            torch.tensor(((1.0, 0.5),)),
            torch.tensor(((0.5, 0.25),)),
            torch.tensor((0.0, 11.764705882352942)),
            identity,
        )
    )
    artifact = torch.load(tmp_path / "temporal-rf.pt", weights_only=True)

    # Then the schema is valid and the estimand/history/cell/context are portable.
    validate_v1_ac_temporal_rf_artifact(artifact)
    assert artifact["identity"]["observed_history_context"].startswith("recorded")
    assert artifact["identity"]["selected_source_image_ids"] == ["image-1"]
    assert artifact["identity"]["cell_order"] == ["cell-1"]


def test_v1_ac_temporal_rf_validator_rejects_missing_identity(tmp_path: Path) -> None:
    # Given a structurally plausible RF artifact without scientific identity.
    artifact = {
        "normal": torch.ones(1, 2),
        "ac_structural_clamp": torch.ones(1, 2),
        "ac_contribution": torch.zeros(1, 2),
        "clamped_ac_pathway": torch.zeros(1, 2),
    }

    # When/Then the artifact contract is checked, missing identity is rejected.
    with pytest.raises(ValueError, match="identity"):
        validate_v1_ac_temporal_rf_artifact(artifact)


def test_v1_ac_runner_writes_validated_no_training_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a four-class V1 checkpoint and deterministic held-out responses.
    import evaluation.mechanistic_retina.karamanlis_v1_ac_perturbation as runner

    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        cell_specific_gains=True,
        lag_steps=2,
        dt_ms=10.0,
    )
    model_config_values = asdict(model_config)
    model_config_values["architecture_mode"] = model_config.architecture_mode.value
    cell_ids = ("on-m", "off-m", "on-p", "off-p")
    cell_types = ("midget", "midget", "parasol", "parasol")
    polarities = ("ON", "OFF", "ON", "OFF")
    edge_index = torch.tensor(((0, 1, 2, 3), (0, 1, 2, 3)))
    positions = torch.zeros(4, 2)
    checkpoint = {
        "schema": "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1",
        "stage": "best_trained",
        "best_step": 4,
        "training_seed": 9,
        "revision": 3,
        "model_config": model_config_values,
        "model": {},
        "cell_ids": cell_ids,
        "cell_types": cell_types,
        "polarities": polarities,
        "edge_index": edge_index,
        "cone_blocks_screen_indices": torch.zeros(4, 4, dtype=torch.long),
        "model_cell_positions": positions,
        "model_cone_positions": positions,
        "cell_positions_um": positions,
        "cone_positions_um": positions,
    }
    checkpoint_path = tmp_path / "model-best.pt"
    torch.save(checkpoint, checkpoint_path)
    cones = torch.tensor([[[0.0] * 4, [1.0] * 4, [1.0] * 4]])
    spikes = torch.zeros(1, 3, 4)
    split = SimpleNamespace(
        cone_drive=cones,
        spike_events=spikes,
        source_image_ids=("held-out",),
        trial_indices=(5,),
    )
    data = SimpleNamespace(
        validation=split,
        dt_ms=10.0,
        cell_ids=cell_ids,
        cell_types=cell_types,
        polarities=polarities,
        edge_index=edge_index,
        cone_blocks_screen_indices=checkpoint["cone_blocks_screen_indices"],
        model_cell_positions=positions,
        model_cone_positions=positions,
        cell_positions_um=positions,
        cone_positions_um=positions,
        pathway_spatial_geometry=object(),
    )
    normal_logits = torch.tensor(
        [[[0.0] * 4, [1.0, -1.0, 2.0, -2.0], [0.5, -0.5, 1.0, -1.0]]]
    )
    clamped_logits = normal_logits * 0.5
    zero_current = torch.zeros_like(normal_logits)
    responses = CollectedResponses(
        ResponseTensors(
            normal_logits,
            torch.sigmoid(normal_logits),
            torch.ones_like(normal_logits),
            torch.ones_like(normal_logits),
        ),
        ResponseTensors(
            clamped_logits,
            torch.sigmoid(clamped_logits),
            zero_current,
            zero_current,
        ),
        True,
    )

    class FakeModel:
        config = model_config

        def load_state_dict(self, _state, *, strict):
            return None

        def eval(self):
            return self

    invariance = {
        "all_parameters_unchanged": True,
        "all_state_tensors_unchanged": True,
        "weights": True,
        "gates": True,
        "tau": True,
        "delay": True,
        "H1": True,
        "BC": True,
        "RGC": True,
        "state_sha256_before": "state",
        "state_sha256_after": "state",
    }
    monkeypatch.setattr(runner, "load_rf_population_geometry", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "load_rf_population_imagesequence", lambda *_a, **_k: data)
    monkeypatch.setattr(runner, "build_mechanistic_retina", lambda *_a, **_k: FakeModel())
    monkeypatch.setattr(runner, "state_snapshot", lambda _model: {})
    monkeypatch.setattr(runner, "parameter_invariance", lambda *_a: {})
    monkeypatch.setattr(runner, "temporal_parameter_invariance", lambda *_a: invariance)
    monkeypatch.setattr(runner, "collect_responses", lambda _request: responses)
    monkeypatch.setattr(
        runner,
        "mean_temporal_rf",
        lambda _request, _indices, clamps: torch.full((4, 2), 0.5 if clamps else 1.0),
    )
    monkeypatch.setattr(
        runner,
        "build_lineage",
        lambda *_a: {
            "checkpoint_sha256": "checkpoint-hash",
            "source_sha256": {"analysis.py": "source-hash"},
        },
    )
    output_dir = tmp_path / "result"

    # When the public V1 runner executes without an optimizer.
    runner.run_v1_ac_perturbation(
        runner.V1ACPerturbationConfig(
            tmp_path,
            tmp_path,
            checkpoint_path,
            output_dir,
        )
    )

    # Then it writes the complete validated bundle and records no training.
    payload = json.loads((output_dir / "results.json").read_text())
    artifact = torch.load(output_dir / "temporal-rf.pt", weights_only=True)
    validate_v1_ac_temporal_rf_artifact(artifact)
    assert payload["execution"]["training_performed"] is False
    assert payload["structural_clamp"]["ac_local_current_exact_zero"] is True
    assert {path.name for path in output_dir.iterdir()} == {
        "responses.pt",
        "results.json",
        "temporal-rf.pt",
    }
