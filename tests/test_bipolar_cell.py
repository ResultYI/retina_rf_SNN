from __future__ import annotations

import pytest
import torch

from models.cells.bipolar import (
    BipolarConfig,
    BipolarConfigurationError,
    BipolarKinetics,
    BipolarLayer,
    BipolarPolarity,
)


def _config() -> BipolarConfig:
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


def test_bipolar_layer_splits_polarity_and_kinetics() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    layer = BipolarLayer(positions, _config())
    modulated_drive = torch.tensor([[1.0, -1.0]])

    # When
    state = layer(modulated_drive)

    # Then
    assert state.output.shape == (1, 2, 2, 2)
    assert state.output[0, BipolarPolarity.ON, BipolarKinetics.SUSTAINED, 0] > 0
    assert state.output[0, BipolarPolarity.ON, BipolarKinetics.SUSTAINED, 1] == 0
    assert state.output[0, BipolarPolarity.OFF, BipolarKinetics.SUSTAINED, 0] == 0
    assert state.output[0, BipolarPolarity.OFF, BipolarKinetics.SUSTAINED, 1] > 0
    assert torch.all(
        state.output[:, :, BipolarKinetics.TRANSIENT]
        >= state.output[:, :, BipolarKinetics.SUSTAINED]
    )
    torch.testing.assert_close(layer.private_source_index, torch.arange(2))


def test_bipolar_layer_accepts_bounded_amacrine_feedback() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    layer = BipolarLayer(positions, _config())
    modulated_drive = torch.tensor([[1.0, -1.0]])
    baseline = layer(modulated_drive)

    # When
    inhibited = layer(
        modulated_drive,
        state=layer.initial_state(1, positions.device),
        amacrine_prev=torch.ones_like(baseline.output),
    )
    inhibited.output.sum().backward()

    # Then
    assert torch.all(inhibited.output <= baseline.output)
    assert torch.all(inhibited.output >= 0)
    assert layer.raw_tau_sustained.grad is not None
    assert layer.raw_tau_transient.grad is not None
    assert layer.raw_g_ab_sustained.grad is not None
    assert layer.raw_g_ab_transient.grad is not None
    assert torch.all(layer.g_ab >= 0)
    assert layer.g_ab[BipolarKinetics.SUSTAINED] < _config().g_ab_sustained_max
    assert layer.g_ab[BipolarKinetics.TRANSIENT] < _config().g_ab_transient_max


def test_bipolar_layer_initializes_state_and_returns_diagnostics() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    layer = BipolarLayer(positions, _config())
    initial_state = layer.initial_state(
        batch_size=1,
        device=positions.device,
        dtype=torch.float64,
    )
    modulated_drive = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

    # When
    state, diagnostics = layer(
        modulated_drive,
        state=initial_state,
        return_diagnostics=True,
    )

    # Then
    assert initial_state.output.shape == (1, 2, 2, 2)
    assert initial_state.transient_baseline.shape == (1, 2, 2)
    assert initial_state.output.dtype == torch.float64
    assert initial_state.transient_baseline.dtype == torch.float64
    torch.testing.assert_close(diagnostics["bipolar_tau_ms"], layer.tau_ms.detach())
    torch.testing.assert_close(diagnostics["bipolar_g_ab"], layer.g_ab.detach())
    torch.testing.assert_close(
        diagnostics["bipolar_silent_fraction"],
        torch.tensor(0.5),
    )
    assert diagnostics["bipolar_on_mean"] > 0
    assert diagnostics["bipolar_off_mean"] > 0
    assert diagnostics["bipolar_sustained_mean"] > 0
    assert diagnostics["bipolar_transient_mean"] > 0
    assert state.output.dtype == torch.float64
    assert all(not value.requires_grad for value in diagnostics.values())


def test_transient_bipolar_adapts_to_constant_drive() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0]])
    layer = BipolarLayer(positions, _config())
    drive = torch.ones((1, 1))
    state = layer.initial_state(1, positions.device)

    # When
    state = layer(drive, state)
    first_transient = state.output[
        0,
        BipolarPolarity.ON,
        BipolarKinetics.TRANSIENT,
        0,
    ]
    for _ in range(199):
        state = layer(drive, state)

    # Then
    final_sustained = state.output[
        0,
        BipolarPolarity.ON,
        BipolarKinetics.SUSTAINED,
        0,
    ]
    final_transient = state.output[
        0,
        BipolarPolarity.ON,
        BipolarKinetics.TRANSIENT,
        0,
    ]
    assert final_sustained > 0.9
    assert final_transient < first_transient
    assert final_transient < 0.05 * final_sustained


def test_bipolar_config_requires_separated_temporal_ranges() -> None:
    # Given / When / Then
    with pytest.raises(BipolarConfigurationError, match="temporal ranges"):
        BipolarConfig(
            dt_ms=5.0,
            initial_tau_sustained_ms=80.0,
            tau_sustained_min_ms=30.0,
            tau_sustained_max_ms=200.0,
            initial_tau_transient_ms=20.0,
            tau_transient_min_ms=5.0,
            tau_transient_max_ms=40.0,
            initial_g_ab_sustained=0.01,
            g_ab_sustained_max=0.1,
            initial_g_ab_transient=0.01,
            g_ab_transient_max=0.3,
        )
