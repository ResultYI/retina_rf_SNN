from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.mechanistic_retina.mechanism_checkpoints import (
    CheckpointIdentity,
    load_final_checkpoint,
    save_final_checkpoint,
)
from evaluation.mechanistic_retina.mechanism_heldout import (
    HeldoutPathway,
    heldout_probes,
)
from evaluation.mechanistic_retina.mechanism_heldout_metrics import (
    HeldoutEvaluationRequest,
    evaluate_heldout,
)
from evaluation.mechanistic_retina.mechanism_identifiability import (
    build_student,
    build_teachers,
)
from evaluation.mechanistic_retina.mechanism_run_types import AblationName
from evaluation.mechanistic_retina.mechanism_runs import ablation_clamps
from evaluation.mechanistic_retina.mechanism_replay_identity import (
    ReplayIdentityError,
    prepare_replay_context,
)
from evaluation.mechanistic_retina.mechanism_replay_runs import (
    replay_metric_comparison,
)
from evaluation.mechanistic_retina.rf_base import CandidateTeacherUsage, load_candidate0
from models.mechanistic_retina.contracts import PathwayClamp
from training.mechanistic_retina.stages import build_seed_data


_CANDIDATE = Path(
    ".omo/evidence/hierarchical-endpoint-and-v4-decision/teacher-preflight-results.json"
)


def _fixture():
    candidate = load_candidate0(
        _CANDIDATE,
        usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        reference_candidate_index=0,
    )
    data = build_seed_data(19, candidate)
    return data, build_teachers(data, candidate)


def _identity(data) -> CheckpointIdentity:
    return CheckpointIdentity(
        architecture_revision="mechanism_identifiable",
        teacher_id="H1-specific",
        teacher_hash="teacher-sha256",
        condition="h1",
        structural_variant=AblationName.NO_H1,
        seed=19,
        step=400,
        run_id="test-run",
        dataset_identity="dataset-sha256",
        cell_order=data.cell_ids,
        cone_order=tuple(range(data.cone_positions.shape[0])),
        lag_order=tuple(range(16)),
        gate_values={"h1": 0.75, "ac_local": 0.0, "ac_transient": 0.0, "history": 0.0},
        config_snapshot={"steps": 400, "seeds": [19, 20, 21]},
        config_hash="config-sha256",
        source_hash="source-sha256",
    )


def test_final_checkpoint_roundtrip_preserves_identity_gates_and_pathway_state(
    tmp_path: Path,
) -> None:
    # Given: a final structural model and complete frozen scientific identity.
    data, _ = _fixture()
    model = build_student(data, 19)
    with torch.no_grad():
        model.gates.set_h1_amplitude_(0.0075)
        model.shared_subunits.raw_connections.add_(0.1)
    identity = _identity(data)
    path = tmp_path / "seed-19" / "no_h1.pt"

    # When: the atomic checkpoint is saved and loaded into a fresh raw model.
    saved = save_final_checkpoint(path, model, identity)
    restored = build_student(data, 20)
    loaded = load_final_checkpoint(path, restored)

    # Then: metadata, gates and the explicit pathway state are exactly restored.
    assert saved.path == path
    assert len(saved.sha256) == 64
    assert loaded == identity
    assert torch.equal(restored.gates.h1, model.gates.h1)
    assert torch.equal(
        restored.shared_subunits.raw_connections,
        model.shared_subunits.raw_connections,
    )


def test_structural_variant_clamps_are_fixed() -> None:
    # Given/When: structural variants are mapped to evaluation clamps.
    no_h1 = ablation_clamps(AblationName.NO_H1)
    no_ac = ablation_clamps(AblationName.NO_AC)

    # Then: the frozen retrained-ablation semantics remain unchanged.
    assert no_h1 == frozenset({PathwayClamp.H1})
    assert no_ac == frozenset(
        {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
    )


def test_heldout_probe_loading_reuses_only_preregistered_inputs() -> None:
    # Given: the unchanged teacher family and preregistered probe definitions.
    data, teachers = _fixture()

    # When: the final H1 and AC held-out subsets are selected.
    h1 = heldout_probes(teachers.h1.model, data, HeldoutPathway.H1)
    ac = heldout_probes(teachers.ac.model, data, HeldoutPathway.AC)

    # Then: exactly the three named frozen probes are loaded for each pathway.
    assert tuple(probe.name for probe in h1) == (
        "diagnostic probe H1-03",
        "diagnostic probe H1-05",
        "diagnostic probe H1-04",
    )
    assert tuple(probe.preregistered_name for probe in h1) == (
        "diagnostic-h1-03",
        "diagnostic-h1-05",
        "diagnostic-h1-04",
    )
    assert tuple(probe.name for probe in ac) == (
        "diagnostic probe AC-05",
        "diagnostic probe AC-04",
        "diagnostic probe AC-03",
    )
    assert tuple(probe.preregistered_name for probe in ac) == (
        "diagnostic-ac-05",
        "diagnostic-ac-04",
        "diagnostic-ac-03",
    )
    assert all(
        probe.stimulus.shape == (1, 32, data.train_cones.shape[-1]) for probe in h1 + ac
    )
    assert all(probe.history.shape == (1, 32, len(data.cell_ids)) for probe in h1 + ac)


def test_heldout_evaluation_is_frozen_finite_and_clamped() -> None:
    # Given: frozen Full and retrained-structural student states.
    data, teachers = _fixture()
    full = build_student(data, 19)
    structural = build_student(data, 19)
    with torch.no_grad():
        full.gates.set_h1_amplitude_(0.0075)
    before = {name: value.detach().clone() for name, value in full.state_dict().items()}

    # When: the H1 held-out evaluator runs without an optimizer.
    result = evaluate_heldout(
        HeldoutEvaluationRequest(
            HeldoutPathway.H1,
            teachers.h1,
            full,
            structural,
            data,
            AblationName.NO_H1,
            19,
        )
    )

    # Then: metrics and mechanism observables are finite and model state is unchanged.
    assert result.optimizer_steps == 0
    assert torch.isfinite(torch.tensor(result.full.expected_ce))
    assert torch.isfinite(torch.tensor(result.structural.expected_ce))
    assert torch.isfinite(torch.tensor(result.clamped.expected_ce))
    assert torch.isfinite(torch.tensor(result.pathway.current))
    assert torch.isfinite(torch.tensor(result.pathway.sensitivity))
    assert torch.isfinite(torch.tensor(result.pathway.rf_cosine))
    assert len(result.responses) == 3
    assert all(
        torch.equal(before[name], value) for name, value in full.state_dict().items()
    )


def test_replay_identity_rejects_prior_evidence_after_contract_revision() -> None:
    # Given: prior evidence predates the explicit failed-teacher usage contract.
    root = Path(__file__).resolve().parents[1]

    # When/Then: the historical identity cannot be silently reused by revision 3.
    with pytest.raises(ReplayIdentityError, match="frozen config differs"):
        prepare_replay_context(root, "identity-test")


def test_replay_comparison_enforces_frozen_numeric_tolerances() -> None:
    # Given: identical old and replay summaries with one controlled CE perturbation.
    previous = {
        "validation_ce": 0.2,
        "gates": {"h1": 1.0, "ac_local": 0.0, "ac_transient": 0.0, "history": 0.0},
        "pathway_cosines": {"BC": 0.0, "H1": 0.99, "AC": 0.0},
        "rf": {"global_cosine": 0.998, "exact_fraction": 1.0},
    }
    matching = dict(previous)
    diverged = dict(previous)
    diverged["validation_ce"] = 0.2001001

    # When: the preregistered replay tolerances are applied.
    accepted = replay_metric_comparison(matching, previous)
    rejected = replay_metric_comparison(diverged, previous)

    # Then: equality passes and CE drift above 1e-4 is rejected.
    assert accepted.passed
    assert accepted.ce_difference == 0.0
    assert not rejected.passed
    assert rejected.ce_difference > 1e-4


def test_final_replay_runner_contract_is_bounded() -> None:
    # Given/When: the final replay runner contract is inspected.
    from scripts.run_mechanism_checkpoint_replay import runner_contract

    contract = runner_contract()

    # Then: watchdog, retry, checkpoint and evidence boundaries are fixed.
    assert contract.monitor_interval_seconds == 300.0
    assert contract.stall_intervals == 2
    assert contract.controlled_retries == 1
    assert contract.required_checkpoints == 15
    assert contract.heldout_optimizer_steps == 0
    assert contract.output_names == (
        "identity-manifest.json",
        "replay-results.json",
        "checkpoint-manifest.json",
        "heldout-h1-results.json",
        "heldout-ac-results.json",
        "false-positive-results.json",
        "per-seed-metrics.csv",
        "runtime-monitor.jsonl",
        "commands.json",
        "decision-report-zh.md",
        "failure.json",
    )
