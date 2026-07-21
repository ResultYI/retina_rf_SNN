from __future__ import annotations

import torch

from data.geometry import local_gaussian_weights, nearest_one_to_one_weights
from models.cells.rgc_runtime import assert_row_stochastic, positions_tensor
from models.cells.rgc_types import RGCConfig, RGCConfigurationError, RGCMosaic


def build_population_pools(
    mosaic: RGCMosaic,
    config: RGCConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bipolar_positions = positions_tensor(
        "bipolar_positions_degs", mosaic.bipolar_positions_degs
    )
    midget_positions = positions_tensor(
        "midget_positions_degs", mosaic.midget_positions_degs
    )
    parasol_positions = positions_tensor(
        "parasol_positions_degs", mosaic.parasol_positions_degs
    )
    if not parasol_positions.shape[0] < midget_positions.shape[0]:
        raise RGCConfigurationError("Expected parasol_count < midget_count")

    private_line = (
        midget_positions.shape == bipolar_positions.shape
        and torch.allclose(midget_positions, bipolar_positions, atol=1e-6)
    )
    midget_pool = (
        nearest_one_to_one_weights(bipolar_positions, midget_positions)
        if private_line
        else local_gaussian_weights(
            bipolar_positions,
            midget_positions,
            config.midget_radius_degs,
            config.midget_sigma_degs,
        )
    ).coalesce()
    parasol_pool = local_gaussian_weights(
        bipolar_positions,
        parasol_positions,
        config.parasol_radius_degs,
        config.parasol_sigma_degs,
    ).coalesce()
    assert_row_stochastic("midget_pool", midget_pool)
    assert_row_stochastic("parasol_pool", parasol_pool)
    return midget_positions, parasol_positions, midget_pool, parasol_pool
