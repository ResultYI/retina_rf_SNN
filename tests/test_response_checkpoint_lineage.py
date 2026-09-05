from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from scripts.run_experiment import _write_parameter_sharing_manifest
from training.response_checkpointing import (
    ResponseCheckpointError,
    inspect_response_checkpoint,
    load_response_checkpoint,
    save_response_checkpoint,
)
from training.response_config import (
    ResponseConfigurationError,
    load_response_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_response_config_rejects_unknown_parameter_sharing_mode(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["model"]["parameter_sharing_mode"] = "permutation_factory"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ResponseConfigurationError, match="parameter_sharing_mode"):
        load_response_config(path)


def test_response_config_rejects_non_string_parameter_sharing_mode(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiment.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["model"]["parameter_sharing_mode"] = ["type_aware"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ResponseConfigurationError, match="parameter_sharing_mode"):
        load_response_config(path)


def test_checkpoint_rejects_revision_one_model_contract(tmp_path: Path) -> None:
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
        checkpoint_kind="last",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["model_contract_revision"] = 1
    torch.save(payload, path)

    with pytest.raises(ResponseCheckpointError, match="Architecture V2"):
        load_response_checkpoint(
            path,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint="dataset",
            target_kind="bernoulli",
            config=config,
        )


def test_stage05_checkpoint_kind_inspects_and_foreign_kind_rejects(
    tmp_path: Path,
) -> None:
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "stage05.pt"
    save_response_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        optimizer_step=0,
        best_nll=float("inf"),
        best_checkpoint_step=0,
        generator=torch.Generator().manual_seed(7),
        fingerprint="dataset",
        target_kind="bernoulli",
        config=config,
        run_id="run-a",
        checkpoint_kind="stage05",
    )

    state = inspect_response_checkpoint(path)

    assert state.optimizer_step == 0
    assert state.checkpoint_kind == "stage05"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["checkpoint_kind"] = "foreign"
    torch.save(payload, path)
    with pytest.raises(ResponseCheckpointError, match="lineage"):
        inspect_response_checkpoint(path)


def test_runner_records_model_parameter_sharing_in_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        '{"evaluation_split": "validation"}',
        encoding="utf-8",
    )
    model = nn.Module()
    rgc = nn.Module()
    rgc.parameter_sharing_mode = "type_blind"
    rgc.matched_initialization = True
    rgc.shuffle_contract = "none"
    rgc.observed_type_labels = ("midget", "parasol")
    rgc.cell_polarities = torch.tensor([0, 1])
    rgc.effective_type_labels = ("pooled", "pooled")
    rgc.parameter_group_labels = ("pooled",)
    rgc.parameter_names = ("threshold",)
    rgc.threshold = lambda: torch.tensor([0.2, 0.2])
    model.rgc = rgc

    _write_parameter_sharing_manifest(tmp_path, model)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sharing = manifest["parameter_sharing"]
    assert sharing | {"initial_effective_parameters": {}} == {
        "mode": "type_blind",
        "matched_initialization": True,
        "shuffle_contract": "none",
        "observed_type_labels": ["midget", "parasol"],
        "cell_polarities": [0, 1],
        "effective_type_labels": ["pooled", "pooled"],
        "parameter_group_labels": ["pooled"],
        "initial_effective_parameters": {},
    }
    assert sharing["initial_effective_parameters"]["threshold"] == pytest.approx(
        [0.2, 0.2]
    )
