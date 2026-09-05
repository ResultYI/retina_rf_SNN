from __future__ import annotations

import torch
from torch import nn

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters


def _model(cell_positions: torch.Tensor):
    cell_count = cell_positions.shape[0]
    return build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            cell_specific_gains=True,
        ),
        torch.tensor(
            [[-0.10, 0.00], [0.00, 0.00], [0.10, 0.00]],
            dtype=torch.float32,
        ),
        cell_positions,
        ("midget",) * cell_count,
        ("ON",) * cell_count,
    )


def test_canonical_disabled_operator_parameters_are_non_trainable() -> None:
    # Given
    model = _model(torch.tensor([[0.00, 0.00]], dtype=torch.float32))

    # When
    operator_parameters = tuple(model.operator.parameters())
    optimizer_parameters = phase1_parameters(model)
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}

    # Then
    assert sum(parameter.numel() for parameter in operator_parameters) == 96
    assert all(not parameter.requires_grad for parameter in operator_parameters)
    assert all(id(parameter) not in optimizer_ids for parameter in operator_parameters)


def test_single_cell_shared_subunit_is_fixed_identity_without_parameter() -> None:
    # Given
    model = _model(torch.tensor([[0.00, 0.00]], dtype=torch.float32))
    features = torch.randn(2, 5, 1, 4, 2, 3)

    # When
    mixed = model.shared_subunits(features)
    named_parameters = dict(model.shared_subunits.named_parameters())

    # Then
    assert "raw_connections" not in named_parameters
    assert not isinstance(model.shared_subunits.raw_connections, nn.Parameter)
    torch.testing.assert_close(model.shared_subunits.connection_matrix(), torch.ones(1, 1))
    assert torch.equal(mixed, features)


def test_multicell_shared_subunit_remains_trainable_and_row_normalized() -> None:
    # Given
    model = _model(
        torch.tensor([[0.00, 0.00], [0.01, 0.00]], dtype=torch.float32)
    )

    # When
    matrix = model.shared_subunits.connection_matrix()
    optimizer_parameters = phase1_parameters(model)

    # Then
    assert isinstance(model.shared_subunits.raw_connections, nn.Parameter)
    assert model.shared_subunits.raw_connections.requires_grad
    assert any(
        parameter is model.shared_subunits.raw_connections
        for parameter in optimizer_parameters
    )
    assert model.shared_subunits.raw_connections.numel() == 4
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(2))


def test_single_cell_v1_parameter_inventory_excludes_fixed_self_edge() -> None:
    # Given
    model = _model(torch.tensor([[0.00, 0.00]], dtype=torch.float32))

    # When
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer_listed = sum(parameter.numel() for parameter in phase1_parameters(model))

    # Then
    assert (total, trainable, optimizer_listed) == (129, 33, 33)


def test_two_cell_v1_parameter_inventory_retains_shared_edges() -> None:
    # Given
    model = _model(
        torch.tensor([[0.00, 0.00], [0.01, 0.00]], dtype=torch.float32)
    )

    # When
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer_listed = sum(parameter.numel() for parameter in phase1_parameters(model))

    # Then
    assert (total, trainable, optimizer_listed) == (136, 40, 40)
