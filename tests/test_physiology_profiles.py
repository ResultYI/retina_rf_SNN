from __future__ import annotations

import numpy as np
import torch

from configs.physiology_profiles import (
    dt_ms_from_time_axis_seconds,
    human_macaque_v1,
)
from models.cells.amacrine import A2AmacrineLayer
from models.cells.bipolar import BipolarLayer
from models.cells.horizontal import H1HorizontalNetwork
from models.cells.rgc import RGCMosaic, RGCPopulationLayer
from models.decoder.local_decoder import DecoderTargets, LocalDecoder
from models.retina_snn import RetinaSNNCore


def test_human_macaque_profile_builds_current_model_configs() -> None:
    # Given
    time_axis_seconds = np.arange(4, dtype=np.float64) * 0.005
    positions = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
    )
    mosaic = RGCMosaic(
        bipolar_positions_degs=positions,
        midget_positions_degs=positions,
        parasol_positions_degs=torch.tensor([[0.05, 0.0], [0.25, 0.0]]),
        residual_positions_degs=torch.tensor([[0.15, 0.0]]),
    )

    # When
    dt_ms = dt_ms_from_time_axis_seconds(time_axis_seconds)
    profile = human_macaque_v1(dt_ms=dt_ms, horizon_count=3)
    core = RetinaSNNCore(
        H1HorizontalNetwork(positions, profile.h1),
        BipolarLayer(positions, profile.bipolar),
        A2AmacrineLayer(positions, profile.a2),
        RGCPopulationLayer(mosaic, profile.rgc),
    )
    LocalDecoder(
        mosaic,
        DecoderTargets(positions, mosaic.parasol_positions_degs),
        profile.decoder,
    )

    # Then
    assert dt_ms == 5.0
    assert profile.name == "human_macaque_v1"
    assert profile.species_priority == ("human", "macaque", "marmoset")
    assert core.h1._dt_ms == 5.0
    assert core.bipolar._dt_ms == 5.0
    assert core.a2._dt_ms == 5.0
    assert profile.decoder.horizon_count == 3


def test_dt_ms_uses_median_frame_interval_for_stable_time_axis() -> None:
    # Given
    time_axis_seconds = np.asarray([0.0, 0.005, 0.010, 0.01501])

    # When / Then
    assert dt_ms_from_time_axis_seconds(time_axis_seconds) == 5.0
