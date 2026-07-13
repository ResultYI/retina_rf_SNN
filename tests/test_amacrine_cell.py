from __future__ import annotations

import pytest
import torch

import models.cells.amacrine as amacrine_module
from models.cells.amacrine import (
    LocalAmacrineConfig,
    LocalAmacrineConfigurationError,
    LocalAmacrineLayer,
)
from models.cells.bipolar import BipolarConfig, BipolarLayer


def test_public_amacrine_name_describes_generic_local_microcircuit() -> None:
    # Given / When / Then
    assert hasattr(amacrine_module, "LocalAmacrineLayer")
    assert not hasattr(amacrine_module, "A2AmacrineLayer")


def _amacrine_config() -> LocalAmacrineConfig:
    return LocalAmacrineConfig(
        radius_degs=0.15,
        sigma_degs=0.1,
        dt_ms=5.0,
        initial_tau_sustained_ms=100.0,
        tau_sustained_min_ms=40.0,
        tau_sustained_max_ms=250.0,
        initial_tau_transient_ms=40.0,
        tau_transient_min_ms=15.0,
        tau_transient_max_ms=100.0,
        initial_g_ba_sustained=0.03,
        g_ba_sustained_max=0.3,
        initial_g_ba_transient=0.05,
        g_ba_transient_max=0.5,
    )


def _bipolar_config() -> BipolarConfig:
    return BipolarConfig(
        dt_ms=5.0,
        initial_tau_sustained_ms=80.0,
        tau_sustained_min_ms=60.0,
        tau_sustained_max_ms=200.0,
        initial_tau_transient_ms=20.0,
        tau_transient_min_ms=5.0,
        tau_transient_max_ms=40.0,
        initial_g_ab_sustained=0.01,
        g_ab_sustained_max=0.1,
        initial_g_ab_transient=0.01,
        g_ab_transient_max=0.3,
    )


def test_local_amacrine_uses_normalized_pool_and_bounded_dynamics() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    layer = LocalAmacrineLayer(positions, _amacrine_config())
    bipolar_output = torch.zeros((1, 2, 2, 3))
    bipolar_output[..., 1] = 1.0

    # When
    state, diagnostics = layer(bipolar_output, return_diagnostics=True)
    state.sum().backward()

    # Then
    row_sums = torch.sparse.sum(layer.spatial_pool, dim=1).to_dense()
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
    assert state.shape == bipolar_output.shape
    assert torch.all(state >= 0)
    assert torch.all(state[..., 0] > 0)
    assert torch.all(state[..., 2] > 0)
    assert torch.all(layer.g_ba > 0)
    assert torch.all(layer.g_ba < torch.tensor([0.3, 0.5]))
    assert layer.raw_tau_sustained.grad is not None
    assert layer.raw_tau_transient.grad is not None
    assert layer.raw_g_ba_sustained.grad is not None
    assert layer.raw_g_ba_transient.grad is not None
    assert (
        diagnostics["amacrine_pooled_mean_abs"]
        >= diagnostics["amacrine_pooled_mean"]
    )
    assert diagnostics["amacrine_state_mean_abs"] >= diagnostics["amacrine_state_mean"]
    assert 0 < diagnostics["amacrine_self_weight_mean"] <= 1
    assert (
        diagnostics["amacrine_self_weight_mean"]
        <= diagnostics["amacrine_self_weight_max"]
    )
    assert diagnostics["amacrine_mean_neighbor_count"] > 1
    assert all(not value.requires_grad for value in diagnostics.values())


def test_local_amacrine_rejects_nonfinite_input_and_previous_state() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0]])
    layer = LocalAmacrineLayer(positions, _amacrine_config())
    valid = torch.zeros((1, 2, 2, 1))

    # When / Then
    with pytest.raises(LocalAmacrineConfigurationError, match="bipolar_output"):
        layer(torch.full_like(valid, torch.nan))
    with pytest.raises(LocalAmacrineConfigurationError, match="previous state"):
        layer(valid, amacrine_prev=torch.full_like(valid, torch.inf))


def test_local_amacrine_state_recovers_after_a_pulse() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0]])
    layer = LocalAmacrineLayer(positions, _amacrine_config())
    pulse = torch.ones((1, 2, 2, 1))

    # When
    state = layer(pulse)
    peak = state.clone()
    for _ in range(200):
        state = layer(torch.zeros_like(pulse), state)

    # Then
    assert torch.all(peak > 0)
    assert torch.all(state >= 0)
    assert torch.all(state < 0.01 * peak)


def test_local_amacrine_state_inhibits_the_next_bipolar_step() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    bipolar = BipolarLayer(positions, _bipolar_config())
    amacrine = LocalAmacrineLayer(positions, _amacrine_config())
    cone_drive = torch.tensor([[1.0, -1.0]])
    bipolar_state = bipolar(cone_drive)
    amacrine_state = amacrine(bipolar_state.output)

    # When
    uninhibited = bipolar(
        cone_drive,
        state=bipolar_state,
        amacrine_prev=torch.zeros_like(amacrine_state),
    )
    inhibited = bipolar(
        cone_drive,
        state=bipolar_state,
        amacrine_prev=amacrine_state,
    )

    # Then
    assert torch.all(inhibited.output <= uninhibited.output)
    assert torch.any(inhibited.output < uninhibited.output)


def test_local_amacrine_handles_pool_without_diagonal_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    off_diagonal_pool = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.ones(2),
        (2, 2),
    ).coalesce()
    monkeypatch.setattr(
        amacrine_module,
        "local_gaussian_weights",
        lambda *_args, **_kwargs: off_diagonal_pool,
    )

    # When
    layer = LocalAmacrineLayer(torch.zeros(2, 2), _amacrine_config())

    # Then
    assert layer.self_weight_mean.item() == 0.0
    assert layer.self_weight_max.item() == 0.0
