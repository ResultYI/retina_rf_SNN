from __future__ import annotations

import inspect

import pytest
import torch

from models.cells.rgc import RGCMosaic, RGCOutput, RGCPopulationTensors
from models.decoder.local_decoder import (
    DecoderTargets,
    LocalDecoder,
    LocalDecoderConfig,
)


def _mosaic() -> RGCMosaic:
    midget = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )
    return RGCMosaic(
        bipolar_positions_degs=midget,
        midget_positions_degs=midget,
        parasol_positions_degs=torch.tensor([[0.05, 0.0], [0.25, 0.0]]),
        residual_positions_degs=torch.tensor([[0.15, 0.0]]),
    )


def _decoder() -> LocalDecoder:
    mosaic = _mosaic()
    return LocalDecoder(
        mosaic,
        DecoderTargets(
            fine_positions_degs=mosaic.midget_positions_degs,
            coarse_positions_degs=mosaic.parasol_positions_degs,
        ),
        LocalDecoderConfig(
            horizon_count=3,
            fine_radius_degs=0.11,
            fine_sigma_degs=0.08,
            coarse_radius_degs=0.21,
            coarse_sigma_degs=0.12,
            residual_weight_max=0.1,
        ),
    )


def _rgc_output(*prefix: int) -> RGCOutput:
    populations = RGCPopulationTensors(
        midget=torch.ones((*prefix, 2, 4)),
        parasol=torch.ones((*prefix, 2, 2)),
        residual=torch.ones((*prefix, 2, 1)),
    )
    return RGCOutput(
        spikes=RGCPopulationTensors(
            midget=torch.zeros_like(populations.midget),
            parasol=torch.zeros_like(populations.parasol),
            residual=torch.zeros_like(populations.residual),
        ),
        rates=populations,
    )


def test_local_decoder_uses_sparse_local_masks_and_bounds_residual_weights() -> None:
    # Given
    decoder = _decoder()
    projections = (
        decoder.fine_midget,
        decoder.fine_parasol,
        decoder.fine_residual,
        decoder.coarse_midget,
        decoder.coarse_parasol,
        decoder.coarse_residual,
    )

    # When
    with torch.no_grad():
        decoder.fine_residual.raw_weight.fill_(100.0)
        decoder.coarse_residual.raw_weight.fill_(-100.0)

    # Then
    assert all(projection.local_mask.layout == torch.sparse_coo for projection in projections)
    assert all("local_mask" in dict(projection.named_buffers()) for projection in projections)
    assert all(projection.raw_weight.shape == (3, 2) for projection in projections)
    assert all(
        projection.local_mask.shape[1] == projection.source_count
        for projection in projections
    )
    assert decoder.fine_residual.effective_weight.abs().max() <= 0.1
    assert decoder.coarse_residual.effective_weight.abs().max() <= 0.1
    assert decoder.residual_weight_penalty() > 0


def test_local_decoder_outputs_fine_and_coarse_predictions_for_step_and_sequence() -> None:
    # Given
    decoder = _decoder()

    # When
    step_output, diagnostics = decoder(_rgc_output(2), return_diagnostics=True)
    sequence_output = decoder(_rgc_output(2, 4))

    # Then
    assert step_output.target_fine.shape == (2, 3, 4)
    assert step_output.target_coarse.shape == (2, 3, 2)
    assert sequence_output.target_fine.shape == (2, 4, 3, 4)
    assert sequence_output.target_coarse.shape == (2, 4, 3, 2)
    assert diagnostics["decoder_midget_weight_norm"] >= 0
    assert diagnostics["decoder_parasol_weight_norm"] >= 0
    assert diagnostics["decoder_residual_weight_norm"] >= 0
    assert diagnostics["decoder_prediction_fine_std"] >= 0
    assert diagnostics["decoder_prediction_coarse_std"] >= 0
    assert all(not value.requires_grad for value in diagnostics.values())


def test_local_decoder_public_input_is_only_rgc_output() -> None:
    # Given
    parameters = inspect.signature(LocalDecoder.forward).parameters

    # When
    public_inputs = tuple(parameters)

    # Then
    assert public_inputs == ("self", "rgc_output", "return_diagnostics")
    assert parameters["rgc_output"].annotation in {"RGCOutput", RGCOutput}


def test_local_decoder_rejects_population_shape_mismatch() -> None:
    # Given
    decoder = _decoder()
    bad_output = _rgc_output(2)
    bad_output = RGCOutput(
        spikes=bad_output.spikes,
        rates=RGCPopulationTensors(
            midget=bad_output.rates.midget[..., :-1],
            parasol=bad_output.rates.parasol,
            residual=bad_output.rates.residual,
        ),
    )

    # When / Then
    with pytest.raises(ValueError, match="midget"):
        decoder(bad_output)
