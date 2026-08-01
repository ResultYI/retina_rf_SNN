from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from evaluation.parameter_audit import ParameterAuditContext, audit_parameter_deltas
from training.response_trainer import _configure_cell_residual_learning


def test_parameter_delta_audit_records_grouped_nonzero_updates() -> None:
    # Given
    initialized = nn.ModuleDict({"rgc": nn.Linear(2, 2)})
    with torch.no_grad():
        initialized["rgc"].bias.zero_()
    trained = copy.deepcopy(initialized)
    with torch.no_grad():
        trained["rgc"].weight.add_(1.0)

    # When
    audit = audit_parameter_deltas(trained, initialized)

    # Then
    by_name = {entry.name: entry for entry in audit}
    assert by_name["rgc.weight"].group == "rgc"
    assert by_name["rgc.weight"].absolute_norm > 0
    assert by_name["rgc.bias"].absolute_norm == 0
    assert by_name["rgc.bias"].relative_norm is None


def test_parameter_delta_audit_labels_adaptation_attribution_roles() -> None:
    # Given
    initialized = _attribution_model()
    trained = copy.deepcopy(initialized)
    with torch.no_grad():
        trained["rgc"]["adaptation_gain"].type_base_raw.add_(
            torch.tensor([-0.1, 0.2])
        )
        trained["rgc"]["adaptation_gain"].cell_residual_raw.add_(0.3)
        trained["bipolar"].raw_polarity_gain.add_(torch.tensor([0.4, -0.5]))

    # When
    audit = audit_parameter_deltas(
        trained,
        initialized,
        ParameterAuditContext(
            type_ids=("midget", "parasol"),
            cell_ids=("on-midget", "off-midget", "on-parasol", "off-parasol"),
        ),
    )

    # Then
    by_name = {entry.name: entry for entry in audit}
    type_base = by_name["rgc.adaptation_gain.type_base_raw"]
    residual = by_name["rgc.adaptation_gain.cell_residual_raw"]
    polarity = by_name["bipolar.raw_polarity_gain"]
    assert type_base.role == "rgc_type_base"
    assert type_base.element_labels == ("midget", "parasol")
    assert type_base.delta_values == pytest.approx((-0.1, 0.2))
    assert residual.role == "rgc_cell_residual"
    assert residual.element_labels == (
        "on-midget",
        "off-midget",
        "on-parasol",
        "off-parasol",
    )
    assert polarity.role == "polarity_pathway"
    assert polarity.element_labels == ("ON", "OFF")


def test_cell_residual_learning_can_be_frozen_for_ablation() -> None:
    # Given
    model = _attribution_model()

    # When
    _configure_cell_residual_learning(model, learnable=False)

    # Then
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.endswith("cell_residual_raw")
    )
    assert model["rgc"]["adaptation_gain"].type_base_raw.requires_grad


def _attribution_model() -> nn.ModuleDict:
    rgc_parameter = nn.Module()
    rgc_parameter.type_base_raw = nn.Parameter(torch.zeros(2))
    rgc_parameter.cell_residual_raw = nn.Parameter(torch.zeros(4))
    bipolar = nn.Module()
    bipolar.raw_polarity_gain = nn.Parameter(torch.zeros(2))
    return nn.ModuleDict(
        {
            "rgc": nn.ModuleDict({"adaptation_gain": rgc_parameter}),
            "bipolar": bipolar,
        }
    )
