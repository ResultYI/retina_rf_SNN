from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from loss.retina import RetinaLosses
from models.cells.rgc_types import RGCOutput
from training.augmentation import AugmentedClip
from training.checkpointing import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_REVISION,
    CheckpointError,
    load_checkpoint,
)
from training.config import ConfigurationError, load_config
from training.state import EnergyBudgetState
from training.trainer import RetinaTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_time_and_checkpoint_contract() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    assert config.data.sequence_steps == 320
    assert config.training.burn_in_steps == 64
    assert config.training.differentiable_steps == 256
    assert config.training.supervised_steps == 96
    assert config.training.checkpoint_block_steps == 32
    assert config.training.batch_size == 4
    assert config.objective.variance_weight > 0
    assert config.objective.homeostasis_weight > 0
    assert config.objective.phenotype_repulsion_weight > 0
    assert CHECKPOINT_SCHEMA == "retina_rf_snn"
    assert CHECKPOINT_SCHEMA_REVISION == 3


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["training"]["unexpected"] = 1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown keys"):
        load_config(path)


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
    from scripts import run_experiment

    config = load_config(ROOT / "configs" / "experiment.yaml")
    args = run_experiment._parse_args(["--stop-after-steps", "160"])
    execution_limit = run_experiment._execution_limit(
        config.training.max_optimizer_steps,
        args.stop_after_steps,
    )

    assert args.stop_after_steps == 160
    assert execution_limit == 160
    assert config.training.max_optimizer_steps == 6000


def test_trainer_constructs_with_true_batch_configuration() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    model = torch.nn.Linear(1, 1)
    decoder = torch.nn.Linear(1, 1)

    trainer = RetinaTrainer(model, decoder, object(), config, 1.0)

    assert trainer.config.training.batch_size == 4


def test_optimizer_step_uses_one_true_batch() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tau = torch.nn.Parameter(torch.tensor(1.0))

    model = TinyModel()
    decoder = torch.nn.Linear(1, 1, bias=False)
    trainer = RetinaTrainer(model, decoder, object(), config, 1.0)
    observed_batch_sizes: list[int] = []

    def fake_forward(
        noisy_input: torch.Tensor,
        clean_target: torch.Tensor,
        *,
        checkpointed: bool,
    ) -> tuple[RetinaLosses, RGCOutput, None]:
        del clean_target, checkpointed
        observed_batch_sizes.append(noisy_input.shape[0])
        total = model.tau.square() + decoder.weight.square().sum()
        zero = total * 0.0
        output_tensor = torch.zeros(noisy_input.shape[0], 1, 2, 1)
        output = RGCOutput(
            hard_spikes=output_tensor,
            surrogate_spikes=output_tensor,
            spike_probability=output_tensor,
            rates=output_tensor,
            generator_potential=output_tensor,
        )
        losses = RetinaLosses(
            total=total,
            reconstruction=total,
            normalized_reconstruction=total,
            energy=zero,
            budget_energy=zero,
            energy_penalty=zero,
            energy_violation=zero,
            wiring=zero,
            variance_floor=zero,
            phenotype_repulsion=zero,
            homeostasis=zero,
        )
        return losses, output, None

    trainer.forward_clip = fake_forward  # type: ignore[method-assign]
    clip = AugmentedClip(
        noisy_input=torch.zeros(1, 2, 1),
        clean_target=torch.zeros(1, 2, 1),
        metadata={"source_id": "test"},
    )

    result = trainer.train_optimizer_step((clip,) * config.training.batch_size)

    assert observed_batch_sizes == [config.training.batch_size]
    assert result.gradient_norm > 0
    assert result.temporal_gradient_norm > 0
