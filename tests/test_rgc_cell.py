from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.cells.bipolar import BipolarKinetics
from models.cells.rgc import (
    RGCConfig,
    RGCConfigurationError,
    RGCMosaic,
    RGCPopulationLayer,
)
from models.cells.rgc_runtime import assert_row_stochastic


def _mosaic() -> RGCMosaic:
    bipolar = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )
    return RGCMosaic(
        bipolar_positions_degs=bipolar,
        midget_positions_degs=bipolar.clone(),
        parasol_positions_degs=torch.tensor([[0.05, 0.0], [0.25, 0.0]]),
        residual_positions_degs=torch.tensor([[0.15, 0.0]]),
    )


def _config() -> RGCConfig:
    return RGCConfig(
        parasol_radius_degs=0.16,
        parasol_sigma_degs=0.1,
        residual_radius_degs=0.25,
        residual_sigma_degs=0.12,
        dt_ms=5.0,
        membrane_tau_ms=20.0,
        adaptation_tau_ms=80.0,
        rate_tau_ms=50.0,
        threshold=0.2,
        surrogate_slope=5.0,
        adaptation_strength=0.1,
        initial_g_ag_midget=0.01,
        g_ag_midget_max=0.1,
        initial_g_ag_parasol=0.03,
        g_ag_parasol_max=0.3,
        initial_g_ag_residual=0.01,
        g_ag_residual_max=0.1,
        residual_drive_scale=0.25,
    )


def test_rgc_populations_use_local_masks_and_are_trainable() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    bipolar_output = torch.ones((2, 2, 2, 4), requires_grad=True)
    amacrine_output = torch.full_like(bipolar_output, 0.1)

    # When
    output, state, diagnostics = layer(
        bipolar_output,
        amacrine_output,
        return_diagnostics=True,
    )
    (
        output.rates.midget.sum()
        + output.rates.parasol.sum()
        + output.rates.residual.sum()
    ).backward()

    # Then
    assert output.spikes.midget.shape == (2, 2, 4)
    assert output.spikes.parasol.shape == (2, 2, 2)
    assert output.spikes.residual.shape == (2, 2, 1)
    assert state.adaptation.midget.shape == output.spikes.midget.shape
    assert layer.midget_positions_degs.shape == (4, 2)
    assert layer.parasol_positions_degs.shape == (2, 2)
    assert layer.residual_positions_degs.shape == (1, 2)
    torch.testing.assert_close(layer.midget_pool.to_dense(), torch.eye(4))
    for pool in (layer.parasol_pool, layer.residual_pool):
        row_sums = torch.sparse.sum(pool, dim=1).to_dense()
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
    assert bipolar_output.grad is not None
    assert layer.raw_g_ag_midget.grad is not None
    assert layer.raw_g_ag_parasol.grad is not None
    assert layer.raw_g_ag_residual.grad is not None
    assert all(not value.requires_grad for value in diagnostics.values())


def test_rgc_debug_checks_reject_nonfinite_previous_state() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    bipolar_output = torch.zeros((1, 2, 2, 4))
    state = layer.initial_state(1, bipolar_output.device)
    bad_membrane = replace(
        state.membrane,
        midget=torch.full_like(state.membrane.midget, torch.inf),
    )
    bad_state = replace(state, membrane=bad_membrane)

    # When / Then
    with pytest.raises(RGCConfigurationError, match="NaN or inf"):
        layer(
            bipolar_output,
            torch.zeros_like(bipolar_output),
            rgc_prev=bad_state,
        )


def test_rgc_midget_and_parasol_use_sustained_and_transient_channels() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    bipolar_output = torch.zeros((1, 2, 2, 4))
    bipolar_output[:, :, BipolarKinetics.SUSTAINED] = 1.0
    amacrine_output = torch.zeros_like(bipolar_output)

    # When
    output, _state = layer(bipolar_output, amacrine_output)

    # Then
    assert torch.all(output.spikes.midget == 1)
    assert torch.all(output.spikes.parasol == 0)
    assert _config().routing_mode == "hard_v1_simplification"


def test_rgc_a2_input_suppresses_population_spikes() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    bipolar_output = torch.ones((1, 2, 2, 4))

    # When
    uninhibited, _ = layer(bipolar_output, torch.zeros_like(bipolar_output))
    inhibited, _ = layer(
        bipolar_output,
        torch.full_like(bipolar_output, 100.0),
    )

    # Then
    assert torch.all(uninhibited.spikes.midget == 1)
    assert torch.all(uninhibited.spikes.parasol == 1)
    assert torch.all(inhibited.spikes.midget == 0)
    assert torch.all(inhibited.spikes.parasol == 0)


def test_rgc_rate_history_and_adaptation_recover_after_pulse() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    pulse = torch.ones((1, 2, 2, 4))
    amacrine_output = torch.zeros_like(pulse)

    # When
    output, state = layer(pulse, amacrine_output)
    pulse_rate = output.rates.midget.clone()
    pulse_adaptation = state.adaptation.midget.clone()
    output, state = layer(
        torch.zeros_like(pulse),
        amacrine_output,
        rgc_prev=state,
    )
    post_pulse_rate = output.rates.midget.clone()
    for _ in range(200):
        output, state = layer(
            torch.zeros_like(pulse),
            amacrine_output,
            rgc_prev=state,
        )

    # Then
    assert torch.all(pulse_rate > 0)
    assert torch.all(pulse_adaptation > 0)
    assert torch.all(post_pulse_rate > 0)
    assert torch.all(output.rates.midget < 0.01 * post_pulse_rate)
    assert torch.all(state.adaptation.midget < pulse_adaptation)


def test_rgc_preserves_signed_a2_current() -> None:
    # Given
    layer = RGCPopulationLayer(_mosaic(), _config())
    bipolar_output = torch.zeros((1, 2, 2, 4))
    amacrine_output = torch.ones_like(bipolar_output)

    # When
    _, state, diagnostics = layer(
        bipolar_output,
        amacrine_output,
        return_diagnostics=True,
    )

    # Then
    assert torch.all(state.membrane.midget < 0)
    assert torch.all(state.membrane.parasol < 0)
    assert torch.all(state.membrane.residual < 0)
    assert diagnostics["rgc_midget_current_negative_fraction"] == pytest.approx(1.0)
    assert diagnostics["rgc_parasol_current_negative_fraction"] == pytest.approx(1.0)
    assert diagnostics["rgc_residual_current_negative_fraction"] == pytest.approx(1.0)
    assert diagnostics["rgc_midget_current_max"] < 0
    assert diagnostics["rgc_parasol_current_max"] < 0
    assert diagnostics["rgc_residual_current_max"] < 0


def test_rgc_rejects_non_normalized_pool() -> None:
    # Given
    weights = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([0.5, 1.0]),
        (2, 2),
    )

    # When / Then
    with pytest.raises(RGCConfigurationError, match="rows must sum to one"):
        assert_row_stochastic("test_pool", weights)
