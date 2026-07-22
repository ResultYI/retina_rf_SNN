from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.checkpointing import CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA_REVISION
from training.config import ConfigurationError, load_config
from training.trainer import EnergyBudgetState


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_time_and_checkpoint_contract() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    assert config.data.sequence_steps == 320
    assert config.training.burn_in_steps == 64
    assert config.training.differentiable_steps == 256
    assert config.training.supervised_steps == 96
    assert config.training.checkpoint_block_steps == 32
    assert CHECKPOINT_SCHEMA == "retina_rf_snn"
    assert CHECKPOINT_SCHEMA_REVISION == 1


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
    assert state.budget is None
    assert state.dual == 0.0
    state.observe(0.2, config.training.reconstruction_bootstrap_steps + 1, config)
    assert state.budget is not None

