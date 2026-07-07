from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import models.cells.horizontal as horizontal_module
from models.cells.horizontal import H1HorizontalConfig, H1HorizontalNetwork


def _config() -> H1HorizontalConfig:
    return H1HorizontalConfig(
        radius_degs=0.16,
        sigma_degs=0.1,
        feedback_radius_degs=0.21,
        feedback_sigma_degs=0.12,
        h1_spacing_degs=0.2,
        dt_ms=5.0,
        initial_tau_ms=50.0,
        tau_min_ms=10.0,
        tau_max_ms=200.0,
        initial_gain=0.01,
        gain_max=0.2,
    )


def _positions() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )


def test_h1_uses_fewer_explicit_nodes_and_bidirectional_local_maps() -> None:
    # Given
    positions = _positions()

    # When
    network = H1HorizontalNetwork(positions, _config())

    # Then
    assert network.h1_positions_degs.shape == (2, 2)
    assert network.cone_to_h1.shape == (2, 4)
    assert network.h1_to_cone.shape == (4, 2)
    assert network.h1_positions_degs.shape[0] < positions.shape[0]
    for weights in (network.cone_to_h1, network.h1_to_cone):
        row_sums = torch.sparse.sum(weights, dim=1).to_dense()
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))


def test_h1_bidirectional_network_has_bounded_damped_dynamics() -> None:
    # Given
    network = H1HorizontalNetwork(_positions(), _config())
    cone_drive = torch.ones((2, 4), dtype=torch.float32)

    # When
    modulated_drive, state = network(cone_drive)
    recovered_drive, recovered_state = network(torch.zeros_like(cone_drive), state)
    (modulated_drive.sum() + recovered_drive.sum()).backward()

    # Then
    expected_state = torch.ones_like(state) * (1.0 - network.temporal_leak)
    surround = torch.sparse.mm(network.h1_to_cone, state.T).T
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(recovered_state, network.temporal_leak * state)
    torch.testing.assert_close(modulated_drive, cone_drive - network.gain * surround)
    assert network.raw_gain.grad is not None
    assert network.raw_tau.grad is not None


def test_h1_initial_state_preserves_shape_device_and_dtype() -> None:
    # Given
    network = H1HorizontalNetwork(_positions(), _config())

    # When
    state = network.initial_state(3, torch.device("cpu"), torch.float64)

    # Then
    assert state.shape == (3, 2)
    assert state.device == torch.device("cpu")
    assert state.dtype == torch.float64
    assert torch.count_nonzero(state) == 0


def test_h1_filters_grid_nodes_without_cone_support() -> None:
    # Given
    axis = torch.arange(0.0, 0.25, 0.05)
    positions = torch.cat(
        (
            torch.stack((axis, torch.zeros_like(axis)), dim=1),
            torch.stack((torch.zeros_like(axis[1:]), axis[1:]), dim=1),
        )
    )
    config = replace(
        _config(),
        radius_degs=0.06,
        sigma_degs=0.03,
        feedback_radius_degs=0.08,
        feedback_sigma_degs=0.04,
        h1_spacing_degs=0.1,
    )

    # When
    network = H1HorizontalNetwork(positions, config)

    # Then
    assert network.h1_positions_degs.shape == (5, 2)
    assert torch.all(
        (network.h1_positions_degs[:, 0] == 0)
        | (network.h1_positions_degs[:, 1] == 0)
    )


def test_h1_debug_checks_reject_nonfinite_input_and_state() -> None:
    # Given
    network = H1HorizontalNetwork(_positions(), _config())
    cone_drive = torch.zeros((1, 4))
    state = network.initial_state(1, cone_drive.device)

    # When / Then
    with pytest.raises(horizontal_module.H1ConfigurationError, match="cone_drive"):
        network(torch.full_like(cone_drive, torch.nan))
    with pytest.raises(horizontal_module.H1ConfigurationError, match="state"):
        network(cone_drive, torch.full_like(state, torch.inf))


def test_h1_returns_detached_bidirectional_diagnostics() -> None:
    # Given
    network = H1HorizontalNetwork(_positions(), _config())
    cone_drive = torch.tensor([[1.0, -0.5, 0.25, 0.75]])

    # When
    modulated_drive, state, diagnostics = network(
        cone_drive,
        return_diagnostics=True,
    )

    # Then
    assert diagnostics["h1_node_count"].item() == 2
    assert diagnostics["h1_node_ratio"].item() == pytest.approx(0.5)
    assert diagnostics["h1_cone_to_h1_mean_neighbor_count"] > 0
    assert diagnostics["h1_h1_to_cone_mean_neighbor_count"] > 0
    assert torch.isfinite(diagnostics["h1_input_surround_corr"])
    assert diagnostics["h1_input_std"] > 0
    assert diagnostics["h1_modulated_std"] > 0
    assert diagnostics["h1_modulated_to_input_std_ratio"] > 0
    torch.testing.assert_close(
        diagnostics["h1_state_max_abs"],
        state.detach().abs().max(),
    )
    torch.testing.assert_close(
        diagnostics["h1_modulated_max_abs"],
        modulated_drive.detach().abs().max(),
    )
    assert all(not value.requires_grad for value in diagnostics.values())


def test_h1_rejects_non_normalized_bidirectional_map(monkeypatch) -> None:
    # Given
    malformed_pool = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([0.5, 1.0]),
        size=(2, 2),
    ).coalesce()
    monkeypatch.setattr(
        horizontal_module,
        "local_gaussian_weights",
        lambda *_args: malformed_pool,
    )

    # When / Then
    with pytest.raises(horizontal_module.H1ConfigurationError, match="sum to one"):
        H1HorizontalNetwork(_positions(), _config())
