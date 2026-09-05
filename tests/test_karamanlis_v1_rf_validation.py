from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.mechanistic_retina.karamanlis_locality_graph import RFMapGrid
from evaluation.mechanistic_retina.karamanlis_v1_rf_artifacts import (
    RFValidationArtifactError,
    validate_checkpoint_data,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_metrics import (
    compare_population_rfs,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation import (
    RFValidationError,
    validate_v1_checkpoint,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation_math import (
    build_stixel_projection,
    project_cone_rf_to_stixels,
    separable_projection,
)


def test_stixel_projection_uses_exact_screen_pixel_overlap() -> None:
    # Given one five-pixel cone block crossing two five-pixel STA stixels.
    blocks = np.asarray(((1, 6, 1, 6),), dtype=np.int64)

    # When the screen-pixel overlap projection is constructed.
    projection = build_stixel_projection(
        blocks,
        x_centers_px=np.asarray((3.0, 8.0)),
        y_centers_px=np.asarray((3.0, 8.0)),
        stixel_width_px=5,
        stixel_height_px=5,
    )

    # Then the four overlapping stixels receive their exact area fractions.
    pairs = {
        int(index): float(weight)
        for index, weight in zip(
            projection.indices[0], projection.weights[0], strict=True
        )
        if index >= 0
    }
    assert pairs == pytest.approx({0: 0.64, 1: 0.16, 2: 0.16, 3: 0.04})
    assert float(projection.weights[0].sum()) == pytest.approx(1.0)


def test_cone_rf_projection_preserves_total_derivative_mass() -> None:
    # Given two non-overlapping cone blocks and a batched cone-space RF.
    blocks = np.asarray(((0, 5, 0, 5), (5, 10, 5, 10)), dtype=np.int64)
    projection = build_stixel_projection(
        blocks,
        x_centers_px=np.asarray((3.0, 8.0)),
        y_centers_px=np.asarray((3.0, 8.0)),
        stixel_width_px=5,
        stixel_height_px=5,
    )
    cone_rf = torch.tensor([[[2.0, -0.5]]])

    # When the RF is expressed on the empirical STA grid.
    stixel_rf = project_cone_rf_to_stixels(cone_rf, projection)

    # Then the derivative mass and requested grid shape are preserved.
    assert stixel_rf.shape == (1, 1, 2, 2)
    assert float(stixel_rf.sum()) == pytest.approx(float(cone_rf.sum()))


def test_separable_projection_keeps_off_temporal_polarity() -> None:
    # Given an OFF spatiotemporal kernel with a positive spatial footprint.
    temporal = torch.tensor((-1.0, -2.0, -0.5))
    spatial = torch.tensor((0.0, 1.0, 0.5, 0.0))
    kernel = temporal[:, None] * spatial[None]

    # When the model RF is reduced using its own temporal profile.
    result = separable_projection(kernel[None])

    # Then the spatial map is positive-oriented without erasing OFF polarity.
    assert torch.allclose(result.spatial[0], spatial / spatial.sum())
    peak = int(result.temporal[0].abs().argmax())
    assert int(torch.sign(result.temporal[0, peak])) == -1


def test_v1_checkpoint_contract_rejects_pathway_mixture() -> None:
    # Given a checkpoint carrying V2 pathway-mixture gains.
    checkpoint = {
        "schema": "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1",
        "stage": "best_trained",
        "model_config": {
            "cell_specific_gains": False,
            "cell_specific_pathway_mixture": True,
        },
    }

    # When the independent V1 RF contract is checked.
    # Then V2 is rejected before any model evaluation can occur.
    with pytest.raises(RFValidationError, match="aggregate BC/AC gains"):
        validate_v1_checkpoint(checkpoint)


def test_checkpoint_data_contract_rejects_geometry_drift() -> None:
    # Given matching cell metadata but a changed current cone geometry.
    checkpoint = {
        "cell_ids": ("1",),
        "cell_types": ("midget",),
        "polarities": ("ON",),
        "edge_index": torch.tensor(((0,), (0,))),
        "cone_blocks_screen_indices": torch.tensor(((0, 5, 0, 5),)),
        "model_cell_positions": torch.zeros((1, 2)),
        "model_cone_positions": torch.zeros((1, 2)),
        "cell_positions_um": torch.zeros((1, 2)),
        "cone_positions_um": torch.zeros((1, 2)),
    }
    data = SimpleNamespace(
        cell_ids=("1",),
        cell_types=("midget",),
        polarities=("ON",),
        edge_index=torch.tensor(((0,), (0,))),
        cone_blocks_screen_indices=torch.tensor(((0, 5, 0, 5),)),
        model_cell_positions=torch.ones((1, 2)),
        model_cone_positions=torch.zeros((1, 2)),
        cell_positions_um=torch.zeros((1, 2)),
        cone_positions_um=torch.zeros((1, 2)),
    )

    # When checkpoint lineage is checked against the current adapter geometry.
    # Then the drift is rejected before the model is built.
    with pytest.raises(RFValidationArtifactError, match="data contract differ"):
        validate_checkpoint_data(checkpoint, data)


def test_rf_metrics_reject_unknown_polarity() -> None:
    # Given an otherwise shape-valid comparison request with unknown polarity.
    spatial = np.ones((1, 1, 1), dtype=np.float32)
    temporal = np.ones((1, 1), dtype=np.float32)
    grid = RFMapGrid(
        np.asarray((3.0,)),
        np.asarray((3.0,)),
        5,
        5,
        7.5,
        np.asarray((400.0, 300.0)),
    )

    # When the comparison contract parses polarity.
    # Then it fails explicitly instead of treating unknown as OFF.
    with pytest.raises(ValueError, match="polarity"):
        compare_population_rfs(
            cell_ids=("1",),
            cell_types=("midget",),
            polarities=("unknown",),
            model_spatial=spatial,
            model_temporal=temporal,
            empirical_spatial=spatial,
            empirical_temporal=temporal,
            empirical_even_temporal=temporal,
            empirical_odd_temporal=temporal,
            model_lag_ms=np.asarray((0.0,)),
            empirical_lag_ms=np.asarray((0.0,)),
            grid=grid,
        )
