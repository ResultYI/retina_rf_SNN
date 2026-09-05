from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from evaluation.parameter_audit import audit_response_readout
from training.response_checkpointing import (
    ResponseCheckpointError,
    load_response_checkpoint,
    save_response_checkpoint,
)
from training.response_config import load_response_config


class _Readout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.response_bias = nn.Parameter(torch.tensor([0.2, -0.3]))
        self.bipolar_readout_gain = nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        self.amacrine_readout_gain = nn.Parameter(
            torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        )

    def synaptic_gain(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgc = _Readout()


def test_readout_audit_tracks_initial_calibrated_and_trained_direct_gains() -> None:
    # Given
    calibrated = _Model()
    trained = deepcopy(calibrated)
    with torch.no_grad():
        trained.rgc.bipolar_readout_gain.add_(1.0)
        trained.rgc.amacrine_readout_gain.sub_(1.0)

    # When
    audit = audit_response_readout(trained, calibrated)

    # Then
    assert audit.initial_bipolar_readout_gain == ((0.0, 0.0), (0.0, 0.0))
    assert audit.calibrated_bipolar_readout_gain == ((1.0, 2.0), (3.0, 4.0))
    assert audit.trained_bipolar_readout_gain == ((2.0, 3.0), (4.0, 5.0))
    assert audit.initial_amacrine_readout_gain == ((0.0, 0.0), (0.0, 0.0))
    assert audit.calibrated_amacrine_readout_gain == ((5.0, 6.0), (7.0, 8.0))
    assert audit.trained_amacrine_readout_gain == ((4.0, 5.0), (6.0, 7.0))


def test_legacy_checkpoint_missing_stage05_field_is_accepted_only_when_disabled(
    tmp_path: Path,
) -> None:
    # Given
    enabled = load_response_config("configs/synthetic_smoke.yaml")
    disabled = replace(
        enabled,
        training=replace(
            enabled.training,
            stage05_readout_calibration_enabled=False,
        ),
    )
    model = nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_response_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        optimizer_step=1,
        best_nll=0.5,
        best_checkpoint_step=1,
        generator=torch.Generator().manual_seed(7),
        fingerprint="dataset",
        target_kind="bernoulli",
        config=disabled,
        run_id="run-a",
        checkpoint_kind="last",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["config"]["training"]["stage05_readout_calibration_enabled"]
    torch.save(payload, path)

    # When / Then
    load_response_checkpoint(
        path,
        model=model,
        optimizer=None,
        generator=None,
        fingerprint="dataset",
        target_kind="bernoulli",
        config=disabled,
    )
    with pytest.raises(ResponseCheckpointError, match="configuration"):
        load_response_checkpoint(
            path,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint="dataset",
            target_kind="bernoulli",
            config=enabled,
        )
