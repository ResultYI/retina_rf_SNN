from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from evaluation.parameter_audit import (
    ParameterAuditContext,
    audit_parameter_deltas,
    audit_response_readout,
)
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


def test_response_readout_audit_separates_initial_calibrated_and_trained() -> None:
    # Given
    calibrated = _readout_model((0.2, -0.3), (1.0, 1.0))
    trained = _readout_model((0.4, -0.1), (1.5, 0.5))

    # When
    audit = audit_response_readout(trained, calibrated)

    # Then
    assert audit.initial_response_bias == (0.0, 0.0)
    assert audit.calibrated_response_bias == pytest.approx((0.2, -0.3))
    assert audit.trained_response_bias == pytest.approx((0.4, -0.1))
    assert audit.initial_effective_synaptic_gain == (1.0, 1.0)
    assert audit.calibrated_effective_synaptic_gain == pytest.approx((1.0, 1.0))
    assert audit.trained_effective_synaptic_gain == pytest.approx((1.5, 0.5))


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


def test_parameter_delta_audit_prefers_model_effective_group_labels() -> None:
    # Given
    initialized = _attribution_model()
    trained = copy.deepcopy(initialized)
    trained["rgc"].parameter_group_labels = ("pooled",)
    with torch.no_grad():
        trained["rgc"]["adaptation_gain"].type_base_raw = nn.Parameter(torch.zeros(1))
        initialized["rgc"]["adaptation_gain"].type_base_raw = nn.Parameter(torch.zeros(1))

    # When
    audit = audit_parameter_deltas(
        trained,
        initialized,
        ParameterAuditContext(type_ids=("midget", "parasol"), cell_ids=("cell",)),
    )

    # Then
    by_name = {entry.name: entry for entry in audit}
    assert by_name["rgc.adaptation_gain.type_base_raw"].element_labels == ("pooled",)


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


def _readout_model(
    bias: tuple[float, float],
    gain: tuple[float, float],
) -> nn.Module:
    model = nn.Module()
    rgc = nn.Module()
    rgc.response_bias = nn.Parameter(torch.tensor(bias))
    rgc.synaptic_gain = lambda: torch.tensor(gain)
    model.rgc = rgc
    return model
