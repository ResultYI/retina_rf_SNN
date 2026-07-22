from __future__ import annotations

import torch

from configs.physiology_profiles import human_macaque
from models.cells.rgc_types import RGCConfig
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel, build_retina_model


def _model() -> RetinaModel:
    positions = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
    )
    profile = human_macaque(dt_ms=10.0, cone_spacing_deg=0.1, eccentricity_deg=4.0)
    config = RGCConfig(
        units_per_center=2,
        support_radius_degs=0.36,
        sigma_min_degs=0.025,
        sigma_initial_degs=0.12,
        sigma_max_degs=0.36,
        dt_ms=10.0,
        readout_rate_tau_ms=50.0,
        max_tau_ms=250.0,
        surrogate_slope=5.0,
    )
    return build_retina_model(positions, profile, config)


def test_single_pool_and_tied_decoder_shapes() -> None:
    torch.manual_seed(1)
    model = _model()
    spatial_weights = model.rgc.compute_spatial_weights()
    output, _ = model.forward_sequence(
        torch.randn(2, 6, 4), spatial_weights=spatial_weights
    )
    assert output.rates.shape == (2, 6, 2, 8)
    assert output.spike_probability.shape == output.rates.shape
    assert spatial_weights.shape == (8, 4)
    decoder = TiedLocalDecoder(8, 4)
    assert decoder(output.rates, spatial_weights).shape == (2, 6, 4)


def test_rgc_parameters_are_per_unit_and_readout_tau_is_fixed() -> None:
    model = _model()
    unit_count = model.rgc.unit_count
    values = (
        model.rgc.spatial_sigma,
        model.rgc.sustained_mix,
        model.rgc.membrane_tau_ms,
        model.rgc.adaptation_tau_ms,
        model.rgc.adaptation_gain,
        model.rgc.amacrine_gain,
        model.rgc.threshold,
        model.rgc.subunit_tau_ms,
        model.rgc.subunit_gain,
    )
    assert all(value.shape == (unit_count,) for value in values)
    assert "readout_rate_tau_ms" not in dict(model.rgc.named_parameters())
