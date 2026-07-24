from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from training.response_checkpointing import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_REVISION,
)
from training.response_config import (
    ResponseConfigurationError,
    load_response_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_response_training_contract() -> None:
    config = load_response_config(ROOT / "configs" / "experiment.yaml")

    assert config.training.burn_in_steps == 64
    assert config.training.differentiable_steps == 256
    assert config.training.checkpoint_block_steps == 32
    assert CHECKPOINT_SCHEMA == "retina_rgc_response_snn"
    assert CHECKPOINT_SCHEMA_REVISION == 1


def test_response_config_rejects_reconstruction_keys(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["objective"] = {"energy_budget_ratio": 0.9}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ResponseConfigurationError, match="keys mismatch"):
        load_response_config(path)


def test_canonical_runner_imports() -> None:
    from scripts import run_experiment

    assert callable(run_experiment.main)


@pytest.mark.parametrize(
    "script",
    (
        "scripts/run_experiment.py",
        "scripts/generate_synthetic_response_benchmark.py",
    ),
)
def test_documented_direct_script_entrypoints_import_project_packages(
    script: str,
) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
