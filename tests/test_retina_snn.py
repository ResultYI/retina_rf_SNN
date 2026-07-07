from __future__ import annotations

import torch

from models.cells.amacrine import A2AmacrineConfig, A2AmacrineLayer
from models.cells.bipolar import BipolarConfig, BipolarLayer
from models.cells.horizontal import H1HorizontalConfig, H1HorizontalNetwork
from models.cells.rgc import RGCConfig, RGCMosaic, RGCPopulationLayer
from models.decoder.local_decoder import (
    DecoderTargets,
    LocalDecoder,
    LocalDecoderConfig,
)
from models.retina_snn import RetinaSNNCore, RetinaSNNState, detach_state


def _positions() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )


def _core() -> RetinaSNNCore:
    positions = _positions()
    h1 = H1HorizontalNetwork(
        positions,
        H1HorizontalConfig(
            0.16,
            0.1,
            0.21,
            0.12,
            0.2,
            5.0,
            50.0,
            10.0,
            200.0,
            0.01,
            0.2,
        ),
    )
    bipolar = BipolarLayer(
        positions,
        BipolarConfig(
            5.0,
            80.0,
            60.0,
            200.0,
            20.0,
            5.0,
            40.0,
            0.01,
            0.1,
            0.01,
            0.3,
        ),
    )
    a2 = A2AmacrineLayer(
        positions,
        A2AmacrineConfig(
            0.16,
            0.1,
            5.0,
            100.0,
            40.0,
            250.0,
            40.0,
            15.0,
            100.0,
            0.03,
            0.3,
            0.05,
            0.5,
        ),
    )
    rgc = RGCPopulationLayer(
        RGCMosaic(
            bipolar_positions_degs=positions,
            midget_positions_degs=positions,
            parasol_positions_degs=torch.tensor([[0.05, 0.0], [0.25, 0.0]]),
            residual_positions_degs=torch.tensor([[0.15, 0.0]]),
        ),
        RGCConfig(
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
        ),
    )
    return RetinaSNNCore(h1, bipolar, a2, rgc)


def test_retina_core_step_matches_explicit_b_a_r_update_order() -> None:
    # Given
    core = _core()
    cone_t = torch.tensor([[1.0, 0.5, -0.5, -1.0]])
    state = core.initial_state(1, cone_t.device, cone_t.dtype)

    # When
    output, next_state, diagnostics = core.step(
        cone_t,
        state,
        return_diagnostics=True,
    )
    cone_mod, h1_state = core.h1(cone_t, state.h1)
    bipolar_state = core.bipolar(
        cone_mod,
        state.bipolar,
        amacrine_prev=state.a2,
    )
    a2_state = core.a2(bipolar_state.output, state.a2)
    expected_output, rgc_state = core.rgc(
        bipolar_state.output,
        a2_state,
        state.rgc,
    )

    # Then
    torch.testing.assert_close(output.rates.midget, expected_output.rates.midget)
    torch.testing.assert_close(output.rates.parasol, expected_output.rates.parasol)
    torch.testing.assert_close(output.rates.residual, expected_output.rates.residual)
    torch.testing.assert_close(next_state.h1, h1_state)
    torch.testing.assert_close(next_state.bipolar.output, bipolar_state.output)
    torch.testing.assert_close(next_state.a2, a2_state)
    torch.testing.assert_close(next_state.rgc.rate.midget, rgc_state.rate.midget)
    assert set(diagnostics) == {"h1", "bipolar", "a2", "rgc"}


def test_retina_core_rollout_has_population_histories_without_future_leakage() -> None:
    # Given
    core = _core()
    x_cone = torch.tensor(
        [
            [
                [0.1, 0.2, -0.1, -0.2],
                [0.2, 0.3, -0.2, -0.3],
                [0.3, 0.4, -0.3, -0.4],
                [0.4, 0.5, -0.4, -0.5],
            ]
        ]
    )
    changed_future = x_cone.clone()
    changed_future[:, -1] = 100.0

    # When
    output, state, diagnostics = core.forward_sequence(
        x_cone,
        return_diagnostics=True,
    )
    changed_output, _ = core.forward_sequence(changed_future)

    # Then
    assert output.rates.midget.shape == (1, 4, 2, 4)
    assert output.rates.parasol.shape == (1, 4, 2, 2)
    assert output.rates.residual.shape == (1, 4, 2, 1)
    assert output.spikes.midget.shape == output.rates.midget.shape
    assert state.h1.shape == (1, 2)
    assert len(diagnostics) == 4
    torch.testing.assert_close(
        output.rates.midget[:, :-1],
        changed_output.rates.midget[:, :-1],
    )
    torch.testing.assert_close(
        output.rates.parasol[:, :-1],
        changed_output.rates.parasol[:, :-1],
    )


def test_detach_state_detaches_every_nested_recurrent_tensor() -> None:
    # Given
    core = _core()
    x_cone = torch.randn((1, 3, 4), requires_grad=True)
    _, state = core.forward_sequence(x_cone)

    # When
    detached = detach_state(state)

    # Then
    assert isinstance(detached, RetinaSNNState)
    assert any(tensor.requires_grad for tensor in _state_tensors(state))
    assert all(not tensor.requires_grad for tensor in _state_tensors(detached))
    assert all(tensor.grad_fn is None for tensor in _state_tensors(detached))


def test_core_and_local_decoder_run_one_batch_forward_backward() -> None:
    # Given
    core = _core()
    positions = _positions()
    mosaic = RGCMosaic(
        bipolar_positions_degs=positions,
        midget_positions_degs=positions,
        parasol_positions_degs=core.rgc.parasol_positions_degs,
        residual_positions_degs=core.rgc.residual_positions_degs,
    )
    decoder = LocalDecoder(
        mosaic,
        DecoderTargets(positions, core.rgc.parasol_positions_degs),
        LocalDecoderConfig(3, 0.16, 0.08, 0.21, 0.12, 0.1),
    )
    with torch.no_grad():
        decoder.fine_midget.raw_weight.fill_(0.05)
        decoder.coarse_parasol.raw_weight.fill_(0.05)
    x_cone = torch.randn((2, 4, 4))

    # When
    rgc_history, _ = core.forward_sequence(x_cone)
    prediction = decoder(rgc_history)
    loss = (
        prediction.target_fine.square().mean()
        + prediction.target_coarse.square().mean()
        + 0.01 * decoder.residual_weight_penalty()
    )
    loss.backward()

    # Then
    assert prediction.target_fine.shape == (2, 4, 3, 4)
    assert prediction.target_coarse.shape == (2, 4, 3, 2)
    assert torch.isfinite(loss)
    assert decoder.fine_midget.raw_weight.grad is not None
    assert core.h1.raw_gain.grad is not None


def _state_tensors(state: RetinaSNNState) -> tuple[torch.Tensor, ...]:
    rgc = state.rgc
    return (
        state.h1,
        state.bipolar.output,
        state.bipolar.transient_baseline,
        state.a2,
        rgc.membrane.midget,
        rgc.membrane.parasol,
        rgc.membrane.residual,
        rgc.adaptation.midget,
        rgc.adaptation.parasol,
        rgc.adaptation.residual,
        rgc.rate.midget,
        rgc.rate.parasol,
        rgc.rate.residual,
    )
