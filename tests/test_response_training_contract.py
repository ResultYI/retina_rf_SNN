from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import yaml

from scripts.run_experiment import ResponseExperimentError, _prepare_output
from training.response_checkpointing import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_REVISION,
    ResponseCheckpointError,
    load_response_checkpoint,
    save_response_checkpoint,
)
from training.response_config import (
    ResponseConfigurationError,
    load_response_config,
)


ROOT = Path(__file__).resolve().parents[1]


class _ReducerProbe:
    executed = False


class _MaliciousCheckpoint:
    def __reduce__(self):
        return (setattr, (_ReducerProbe, "executed", True))


def test_canonical_response_training_contract() -> None:
    config = load_response_config(ROOT / "configs" / "experiment.yaml")

    assert config.training.burn_in_steps == 64
    assert config.training.differentiable_steps == 256
    assert config.training.checkpoint_block_steps == 32
    assert CHECKPOINT_SCHEMA == "retina_rgc_response_snn"
    assert CHECKPOINT_SCHEMA_REVISION == 3


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


def test_checkpoint_restores_optimizer_step_rng_and_config(tmp_path: Path) -> None:
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(7)
    path = tmp_path / "checkpoint.pt"
    save_response_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        optimizer_step=3,
        best_nll=0.4,
        best_checkpoint_step=2,
        generator=generator,
        fingerprint="dataset",
        target_kind="bernoulli",
        config=config,
        run_id="run-a",
        checkpoint_kind="last",
    )
    expected_random = torch.rand(3, generator=generator)
    restored_generator = torch.Generator().manual_seed(99)

    state = load_response_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        generator=restored_generator,
        fingerprint="dataset",
        target_kind="bernoulli",
        config=config,
        expected_run_id="run-a",
    )

    assert state.optimizer_step == 3
    assert state.best_nll == 0.4
    assert state.best_checkpoint_step == 2
    assert state.run_id == "run-a"
    assert torch.equal(torch.rand(3, generator=restored_generator), expected_random)


def test_checkpoint_rejects_pickle_payload_without_executing_reducer(
    tmp_path: Path,
) -> None:
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = nn.Linear(2, 1)
    path = tmp_path / "malicious.pt"
    torch.save(_MaliciousCheckpoint(), path)
    _ReducerProbe.executed = False

    with pytest.raises(ResponseCheckpointError):
        load_response_checkpoint(
            path,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint="dataset",
            target_kind="bernoulli",
            config=config,
        )

    assert _ReducerProbe.executed is False


def test_checkpoint_rejects_malformed_safe_payload(tmp_path: Path) -> None:
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = nn.Linear(2, 1)
    path = tmp_path / "malformed.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_revision": CHECKPOINT_SCHEMA_REVISION,
            "model": [],
            "optimizer": {},
            "optimizer_step": 3,
            "best_nll": 0.4,
            "best_checkpoint_step": 2,
            "sampling_rng": torch.Generator().manual_seed(7).get_state(),
            "dataset_fingerprint": "dataset",
            "target_kind": "bernoulli",
            "config": asdict(config),
            "run_id": "run-a",
            "parent_run_id": None,
            "model_contract_revision": 1,
            "checkpoint_kind": "last",
        },
        path,
    )

    with pytest.raises(ResponseCheckpointError, match="invalid"):
        load_response_checkpoint(
            path,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint="dataset",
            target_kind="bernoulli",
            config=config,
        )


def test_checkpoint_rejects_foreign_run_lineage(tmp_path: Path) -> None:
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_response_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        optimizer_step=1,
        best_nll=0.5,
        best_checkpoint_step=1,
        generator=torch.Generator().manual_seed(7),
        fingerprint="dataset",
        target_kind="bernoulli",
        config=config,
        run_id="run-a",
        checkpoint_kind="best",
    )

    with pytest.raises(ResponseCheckpointError, match="lineage"):
        load_response_checkpoint(
            path,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint="dataset",
            target_kind="bernoulli",
            config=config,
            expected_run_id="run-b",
        )


def test_fresh_run_rejects_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint_best_nll.pt").write_bytes(b"stale")
    args = argparse.Namespace(
        output=str(output),
        diagnostics_only=False,
        checkpoint=None,
        resume=None,
        overwrite=False,
    )

    with pytest.raises(ResponseExperimentError, match="empty output"):
        _prepare_output(args)


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
