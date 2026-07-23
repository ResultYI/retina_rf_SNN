from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from training import experiment_cli
from training.checkpointing import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_REVISION,
    CheckpointError,
    load_checkpoint,
)
from training.config import ConfigurationError, load_config
from training.state import EnergyBudgetState


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_time_and_checkpoint_contract() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    assert config.data.sequence_steps == 320
    assert config.training.burn_in_steps == 64
    assert config.training.differentiable_steps == 256
    assert config.training.supervised_steps == 96
    assert config.training.checkpoint_block_steps == 32
    assert config.training.batch_size == 4
    assert config.training.decoder_freeze_steps == 0
    assert config.objective.variance_weight > 0
    assert config.objective.homeostasis_weight > 0
    assert config.objective.phenotype_repulsion_weight > 0
    assert CHECKPOINT_SCHEMA == "retina_rf_snn"
    assert CHECKPOINT_SCHEMA_REVISION == 5


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["training"]["unexpected"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_config(path)


def test_auxiliary_objective_weights_may_be_zero(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key in (
        "wiring_weight",
        "variance_weight",
        "phenotype_repulsion_weight",
        "homeostasis_weight",
    ):
        raw["objective"][key] = 0.0
    path = tmp_path / "zero-weights.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(path)

    assert config.objective.phenotype_repulsion_weight == 0.0


def test_energy_budget_is_inactive_during_bootstrap() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    state = EnergyBudgetState()
    state.observe(0.2, 1, config)
    assert state.current_budget is None
    assert state.dual == 0.0
    state.observe(0.2, config.training.reconstruction_bootstrap_steps + 1, config)
    assert state.current_budget is not None
    assert state.target_budget == pytest.approx(
        state.reference_energy * config.objective.energy_budget_ratio
    )


def test_energy_target_freezes_after_bootstrap() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    state = EnergyBudgetState()
    state.observe(0.2, config.training.reconstruction_bootstrap_steps, config)
    reference = state.reference_energy
    state.observe(0.8, config.training.reconstruction_bootstrap_steps + 1, config)
    target = state.target_budget
    state.observe(1.2, config.training.budget_ramp_end_step, config)
    assert state.reference_energy == reference
    assert state.target_budget == target
    assert state.current_budget == pytest.approx(target)


def test_revision_two_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "revision-two.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "schema_revision": 2}, path)
    with pytest.raises(CheckpointError, match="revision"):
        load_checkpoint(path, torch.device("cpu"))


def test_experiment_runner_imports_without_evaluation_side_effects() -> None:
    from scripts import run_experiment

    assert callable(run_experiment.main)


def test_runner_stop_after_steps_preserves_configured_horizon() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    args = experiment_cli.parse_experiment_args(
        ["--stop-after-steps", "160"]
    )
    execution_limit = experiment_cli.execution_limit(
        config.training.max_optimizer_steps,
        args.stop_after_steps,
    )

    assert args.stop_after_steps == 160
    assert execution_limit == 160
    assert config.training.max_optimizer_steps == 6000


def test_representation_diagnostic_mode_applies_one_canonical_policy() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    args = experiment_cli.parse_experiment_args(
        [
            "--representation-diagnostic-steps",
            "50",
            "--diagnostic-core-lr",
            "0.0004",
        ]
    )

    diagnostic = experiment_cli.apply_invocation_overrides(config, args)

    assert diagnostic.training.core_lr == pytest.approx(4e-4)
    assert diagnostic.training.decoder_freeze_steps == 50
    assert diagnostic.training.validation_interval_steps == 10
    assert diagnostic.training.max_optimizer_steps == 6000
    assert diagnostic.objective.phenotype_repulsion_weight == 0.0
    assert experiment_cli.execution_limit(
        diagnostic.training.max_optimizer_steps,
        args.stop_after_steps,
        args.representation_diagnostic_steps,
    ) == 50


def test_representation_diagnostic_stops_after_three_material_regressions() -> None:
    args = experiment_cli.parse_experiment_args(
        ["--representation-diagnostic-steps", "50"]
    )

    assert experiment_cli.diagnostic_should_stop(
        0.35,
        [0.354, 0.355, 0.356],
        args,
    )
    assert not experiment_cli.diagnostic_should_stop(
        0.35,
        [0.354, 0.349, 0.356],
        args,
    )
