from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from evaluation.mechanistic_retina.clean_sampled_benchmark import (
    CleanBenchmarkConfig,
    build_clean_benchmark,
    run_clean_benchmark,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    effective_parameter_values,
    explicit_delay_bounds,
    explicit_delay_values,
    tau_values,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from models.mechanistic_retina.model import build_mechanistic_retina


def _tiny_config() -> CleanBenchmarkConfig:
    return CleanBenchmarkConfig(
        train_stimuli=4,
        validation_stimuli=2,
        time_steps=20,
        trials=2,
        steps=2,
        checkpoint_steps=(0, 1, 2),
        batch_size=2,
    )


def test_clean_teacher_has_all_pathways_and_student_does_not_copy_weights() -> None:
    state = build_clean_benchmark(_tiny_config())
    teacher_gates = state.teacher.gates.values(frozenset())
    group_local = teacher_gates.ac_local[:4]

    assert set(state.teacher_effects) == {"H1", "direct_BC", "AC"}
    assert all(value > 0 for value in state.teacher_effects.values())
    assert torch.unique(group_local).numel() == 4
    assert not torch.equal(
        state.teacher.bipolar.raw_weights,
        state.student.bipolar.raw_weights,
    )
    assert not hasattr(state.teacher.amacrine, "raw_weights")
    assert not hasattr(state.student.amacrine, "raw_weights")
    assert not torch.equal(state.teacher.gates.h1, state.student.gates.h1)
    assert not torch.equal(state.teacher.h1.raw_delay, state.student.h1.raw_delay)
    assert not torch.equal(
        state.teacher.feature_bank.raw_delay,
        state.student.feature_bank.raw_delay,
    )
    assert not torch.equal(state.teacher.amacrine.raw_delay, state.student.amacrine.raw_delay)
    assert torch.equal(
        state.teacher.shared_subunits.edge_index,
        state.student.shared_subunits.edge_index,
    )
    assert torch.equal(
        state.teacher.feature_bank.polarity_sign,
        state.student.feature_bank.polarity_sign,
    )
    assert torch.equal(
        state.teacher.feature_bank.path_spatial_basis,
        state.student.feature_bank.path_spatial_basis,
    )
    assert state.train_spikes.shape == (4, 2, 20, 8)
    assert state.validation_spikes.shape == (2, 2, 20, 8)
    assert state.teacher.cell_gains is not None
    assert state.student.cell_gains is not None
    assert torch.equal(state.teacher.cell_gains.audit_values, torch.ones(8, 2))
    assert torch.equal(state.student.cell_gains.audit_values, torch.ones(8, 2))


def test_clean_reports_contain_only_shared_bc_encoder_and_downstream_ac_parameters() -> None:
    state = build_clean_benchmark(_tiny_config())
    model = state.student

    effective = effective_parameter_values(model)
    taus = tau_values(model)
    delays = explicit_delay_values(model)

    assert set(effective) == {
        "H1_effective_amplitude", "BC_effective_weights", "AC_effective_gates",
        "cell_BC_gains", "cell_AC_gains",
    }
    assert set(taus) == {
        "H1", "BC_sustained_basis", "BC_transient_basis",
        "AC_local_state", "AC_transient_state",
    }
    assert set(delays) == {
        "H1", "BC_sustained", "BC_transient",
        "AC_local_downstream", "AC_transient_downstream",
    }
    assert set(explicit_delay_bounds(model)) == set(delays)
    torch.testing.assert_close(delays["AC_local_downstream"], model.amacrine.delay_ms[0:1])
    torch.testing.assert_close(delays["AC_transient_downstream"], model.amacrine.delay_ms[1:2])


def test_tiny_clean_benchmark_writes_sampled_only_evidence(tmp_path) -> None:
    result = run_clean_benchmark(_tiny_config(), tmp_path, pathway_diagnostics=True)
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    assert result.training_target == "sampled_rgc_spikes_only"
    assert result.loaded_checkpoint is False
    assert result.tau_parameters_learned is True
    assert result.explicit_pathway_delays_learned is True
    assert payload["model_revision"] == MECHANISTIC_MODEL_REVISION
    assert payload["validation_spike_nll"]["raw"] > 0
    assert payload["validation_spike_nll"]["trained"] > 0
    assert set(payload["rf"]["pathways"]) == {"H1", "BC", "AC"}
    assert set(payload["pathway_updates"]) == {"H1", "BC", "AC"}
    assert all(
        payload["pathway_updates"][name]["gradient_seen"] for name in ("H1", "BC", "AC")
    )
    assert all(
        payload["pathway_updates"][name]["actually_updated"]
        for name in ("H1", "BC", "AC")
    )
    assert set(payload["tau_parameters"]) == {
        "H1",
        "BC_sustained_basis",
        "BC_transient_basis",
        "AC_local_state",
        "AC_transient_state",
    }
    assert all(record["gradient_seen"] for record in payload["tau_parameters"].values())
    assert all(
        record["actually_updated"] for record in payload["tau_parameters"].values()
    )
    assert set(payload["explicit_pathway_delay_parameters"]) == {
        "H1",
        "BC_sustained",
        "BC_transient",
        "AC_local_downstream",
        "AC_transient_downstream",
    }
    assert all(
        record["gradient_seen"]
        for record in payload["explicit_pathway_delay_parameters"].values()
    )
    assert all(
        record["actually_updated"]
        for record in payload["explicit_pathway_delay_parameters"].values()
    )
    assert payload["explicit_pathway_delay_diagnostics"] == {
        "all_finite": True,
        "order_valid": True,
        "any_hit_boundary": False,
    }
    assert set(payload["timing_concepts"]) == {
        "tau",
        "explicit_pathway_delay",
        "rf_lag_window",
        "rgc_history_shift",
    }
    assert payload["rf_lag_window"] == {
        "learnable": False,
        "lag_steps": 16,
        "dt_ms": 5.0,
        "maximum_lag_ms": 75.0,
    }
    assert payload["rgc_history_shift"] == {
        "learnable": False,
        "shift_steps": 1,
        "shift_ms": 5.0,
    }
    assert payload["temporal_rf_change"]["difference_norm"] > 0
    assert payload["parameter_inventory"]["optimizer_listed"] > 0
    recovery = payload["effective_parameter_recovery"]
    assert recovery["comparison_space"] == "effective_normalized_parameters_only"
    assert set(recovery["parameters"]) == {
        "H1_effective_amplitude",
        "BC_effective_weights",
        "AC_effective_gates",
        "cell_BC_gains",
        "cell_AC_gains",
    }
    assert all(
        "raw" not in name.lower() for name in recovery["parameters"]
    )
    assert payload["cell_gain_updates"]["gradient_seen"] is True
    assert payload["cell_gain_updates"]["actually_updated"] is True
    assert set(payload["structural_ablation"]) == {"H1_off", "direct_BC_off", "AC_off"}
    assert all(
        record["structural_current_exact_zero"] is True
        for record in payload["structural_ablation"].values()
    )
    assert (tmp_path / "student-raw.pt").is_file()
    assert (tmp_path / "student-trained.pt").is_file()
    assert (tmp_path / "teacher.pt").is_file()
    assert (tmp_path / "sampled-data.pt").is_file()
    assert (tmp_path / "rf-tensors.pt").is_file()
    assert (tmp_path / "prediction-tensors.pt").is_file()
    assert (tmp_path / "effective-parameter-recovery.pt").is_file()
    assert (tmp_path / "structural-ablation.pt").is_file()
    raw_checkpoint = torch.load(tmp_path / "student-raw.pt", weights_only=False)
    trained_checkpoint = torch.load(tmp_path / "student-trained.pt", weights_only=False)
    raw = raw_checkpoint["model_state"]
    trained = trained_checkpoint["model_state"]
    temporal_keys = (
        "h1.raw_tau",
        "feature_bank.raw_tau",
        "amacrine.raw_tau",
        "h1.raw_delay",
        "feature_bank.raw_delay",
        "amacrine.raw_delay",
    )
    assert all(key in raw for key in temporal_keys)
    assert all(not torch.equal(raw[key], trained[key]) for key in temporal_keys)
    replay = build_mechanistic_retina(
        MechanisticRetinaConfig(**raw_checkpoint["model_config"]),
        raw_checkpoint["cone_positions"],
        raw_checkpoint["cell_positions"],
        raw_checkpoint["cell_types"],
        raw_checkpoint["polarities"],
    )
    replay.load_state_dict(raw)
    with pytest.raises(FileExistsError, match="output directory must be empty"):
        run_clean_benchmark(_tiny_config(), tmp_path)


def test_clean_benchmark_script_imports_from_direct_file_entry() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_clean_sampled_mechanistic_benchmark.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "clean sampled-spike mechanistic benchmark" in completed.stdout
    assert "bounded_differentiable_delay_learning" in completed.stdout
