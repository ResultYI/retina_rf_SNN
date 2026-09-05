from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from evaluation.mechanistic_retina.rf_base import (
    Candidate0Reference,
    CandidateTeacherUsage,
    load_candidate0,
)
from training.mechanistic_retina.stages import build_seed_data


_CANDIDATE = Path(
    ".omo/evidence/hierarchical-endpoint-and-v4-decision/teacher-preflight-results.json"
)


def _data():
    candidate = load_candidate0(
        _CANDIDATE,
        usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        reference_candidate_index=0,
    )
    return candidate, build_seed_data(19, candidate)


def test_three_teachers_have_matched_base_and_distinct_target_mechanisms() -> None:
    # Given: the frozen identities and Candidate0 reference.
    candidate, data = _data()
    from evaluation.mechanistic_retina.mechanism_identifiability import build_teachers

    # When: the three preregistered teachers are constructed.
    teachers = build_teachers(data, candidate)

    # Then: BC identity is shared while H1 and AC effects are finite and distinct.
    assert tuple(teachers) == ("Base", "H1-specific", "AC-specific")
    base = teachers["Base"]
    h1 = teachers["H1-specific"]
    ac = teachers["AC-specific"]
    assert torch.equal(base.bc_rf, h1.bc_rf)
    assert torch.equal(base.bc_rf, ac.bc_rf)
    assert not torch.equal(base.validation_probability, h1.validation_probability)
    assert not torch.equal(base.validation_probability, ac.validation_probability)
    assert not torch.equal(h1.validation_probability, ac.validation_probability)


def test_teacher_preflight_covers_probes_and_heldout_split() -> None:
    # Given: the preregistered teacher family.
    candidate, data = _data()
    from evaluation.mechanistic_retina.mechanism_identifiability import (
        build_teachers,
        teacher_preflight,
    )

    # When: deterministic preflight is evaluated before student training.
    result = teacher_preflight(data, build_teachers(data, candidate))

    # Then: both present teachers pass effect, RF and held-out checks with named probes.
    assert result["H1-specific"].passed
    assert result["AC-specific"].passed
    assert result["H1-specific"].probe_names == (
        "diagnostic-h1-01",
        "diagnostic-h1-02",
        "diagnostic-h1-03",
        "diagnostic-h1-04",
        "diagnostic-h1-05",
    )
    assert result["AC-specific"].probe_names == (
        "diagnostic-ac-01",
        "diagnostic-ac-02",
        "diagnostic-ac-03",
        "diagnostic-ac-04",
        "diagnostic-ac-05",
    )
    assert data.final_test_boundary[-1] == "FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED"


def test_raw_initialization_fixed_schedule_and_no_rf_target_loss() -> None:
    # Given: two raw students and the frozen protocol config.
    _, data = _data()
    from evaluation.mechanistic_retina.mechanism_identifiability import (
        build_student,
        load_mechanism_config,
    )
    from evaluation.mechanistic_retina.structural_ablation import training_contract

    config = load_mechanism_config(Path("configs/mechanism_identifiability.yaml"))
    first = build_student(data, 19)
    second = build_student(data, 19)
    third = build_student(data, 20)

    # When: initial state and optimization contract are inspected.
    first_state = first.state_dict()
    second_state = second.state_dict()
    third_state = third.state_dict()
    contract = training_contract(config)

    # Then: initialization is raw/deterministic, seed-sensitive, fixed-400 and likelihood-only.
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    assert any(not torch.equal(first_state[key], third_state[key]) for key in first_state)
    assert contract.initialization == "teacher-independent-raw"
    assert contract.steps == 400
    assert contract.rf_target_loss_used is False
    assert contract.checkpoint_steps == (0, 50, 100, 200, 400)


def test_checkpoint_roundtrip_includes_gates_and_shared_connections(tmp_path: Path) -> None:
    # Given: a student with projected gates and a changed shared connection parameter.
    _, data = _data()
    from evaluation.mechanistic_retina.mechanism_identifiability import build_student

    model = build_student(data, 19)
    with torch.no_grad():
        model.gates.set_h1_amplitude_(0.0075)
        model.shared_subunits.raw_connections.add_(0.1)
    path = tmp_path / "mechanism.pt"

    # When: state is saved and loaded into a fresh raw instance.
    torch.save(model.state_dict(), path)
    restored = build_student(data, 20)
    restored.load_state_dict(torch.load(path, weights_only=True))

    # Then: gate and shared-subunit state round-trip exactly.
    assert torch.equal(model.gates.h1, restored.gates.h1)
    assert torch.equal(
        model.shared_subunits.raw_connections,
        restored.shared_subunits.raw_connections,
    )


def test_runner_contract_uses_one_entrypoint_watchdog_and_strict_artifacts() -> None:
    # Given: the unique mechanism-identifiability runner.
    from scripts.run_mechanism_identifiability import runner_contract

    # When: its runtime and artifact contract is inspected.
    contract = runner_contract()

    # Then: monitoring is 300s x2 and only frozen evidence names are permitted.
    assert contract.monitor_interval_seconds == 300.0
    assert contract.stall_intervals == 2
    assert contract.controlled_retries == 1
    assert contract.output_names == (
        "identity-manifest.json",
        "experiment-config.yaml",
        "diagnosis-results.json",
        "teacher-preflight-results.json",
        "noise-free-results.json",
        "sampled-confirmation-results.json",
        "per-pathway-metrics.csv",
        "per-cell-metrics.csv",
        "runtime-monitor.jsonl",
        "run.log",
        "failure.json",
        "commands.json",
        "decision-report-zh.md",
    )
