from __future__ import annotations

import torch

from evaluation.residual_ablation import residual_ablation_report
from evaluation.rf_probe import (
    GradientRFRequest,
    RGCPopulationName,
    WhiteNoiseSTARequest,
    gradient_rf,
    white_noise_sta,
)
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from scripts.train_stage1 import _parse_horizons
from training.stage1 import (
    Stage1BuildConfig,
    Stage1OptimizerConfig,
    build_stage1_components,
    build_stage1_optimizer,
)


def _positions() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
         [0.4, 0.0], [0.5, 0.0], [0.6, 0.0], [0.7, 0.0]]
    )


def test_stage1_factory_builds_core_decoder_and_row_normalized_pools() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=3),
    )

    # When
    optimizer = build_stage1_optimizer(
        components.core,
        components.decoder,
        Stage1OptimizerConfig(core_lr=1e-4, decoder_lr=1e-3),
    )

    # Then
    assert components.profile.name == "human_macaque_v1"
    assert components.target_pools.fine.shape == (8, 8)
    assert components.target_pools.coarse.shape[1] == 8
    torch.testing.assert_close(
        torch.sparse.sum(components.target_pools.fine, dim=1).to_dense(),
        torch.ones(8),
    )
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 1e-3


def test_rf_probe_reports_gradient_rf_and_white_noise_sta_shapes() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    x_cone = torch.randn(2, 5, 8)

    # When
    gradient = gradient_rf(
        components.core,
        GradientRFRequest(x_cone, RGCPopulationName.MIDGET, 0, 0),
    )
    sta = white_noise_sta(
        components.core,
        WhiteNoiseSTARequest(8, 5, 4, RGCPopulationName.MIDGET, 0, 0),
    )

    # Then
    assert gradient.gradient.shape == (5, 8)
    assert gradient.response.shape == (2,)
    assert sta.sta.shape == (5, 8)
    assert torch.isfinite(sta.response_mean)


def test_residual_ablation_reports_residual_decoder_contribution() -> None:
    # Given
    components = build_stage1_components(
        _positions(),
        Stage1BuildConfig(dt_ms=5.0, horizon_count=1),
    )
    with torch.no_grad():
        components.decoder.fine_residual.raw_weight.fill_(0.5)
        components.decoder.coarse_residual.raw_weight.fill_(0.5)
    output = RGCOutput(
        spikes=RGCPopulationTensors(
            midget=torch.zeros(1, 2, 8),
            parasol=torch.zeros(1, 2, 2),
            residual=torch.ones(1, 2, 1),
        ),
        rates=RGCPopulationTensors(
            midget=torch.zeros(1, 2, 8),
            parasol=torch.zeros(1, 2, 2),
            residual=torch.ones(1, 2, 1),
        ),
    )

    # When
    report = residual_ablation_report(components.decoder, output)

    # Then
    assert report.fine_residual_contribution > 0
    assert report.coarse_residual_contribution > 0
    assert _parse_horizons("1,2,4") == (1, 2, 4)
