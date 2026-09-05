from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from data.karamanlis_cells import CellSelection
from data.karamanlis_rf_population import (
    RFPopulationAdapterConfig,
    build_rf_pathway_geometry,
    load_rf_population_geometry,
    load_rf_population_imagesequence,
)
from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.real_early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingTrainingRequest,
    fit_real_spike_model_early_stopping,
)
from training.mechanistic_retina.real_sampled import _validation_metrics


_SESSION = Path(
    "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
)
_GRAPH = Path(
    "output/real_data/karamanlis_2024_population_locality_graph_v1"
)


def _split(cones: torch.Tensor, events: torch.Tensor) -> RealSequenceSplit:
    counts = events.to(torch.int64)
    return RealSequenceSplit(
        cone_drive=cones,
        spike_counts=counts,
        spike_events=events,
        valid_mask=torch.ones_like(events, dtype=torch.bool),
        source_image_ids=tuple(f"image-{index}" for index in range(cones.shape[0])),
        trial_indices=tuple(range(cones.shape[0])),
    )


def test_rf_pathway_geometry_uses_full_contours_and_nested_bc_ac_support() -> None:
    # Given: two measured RF contours sampled by four projected cone blocks.
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 1:7, 1:7] = True
    masks[1, 2:8, 2:8] = True
    cone_blocks = np.asarray(
        ((0, 4, 0, 4), (0, 4, 4, 8), (4, 8, 0, 4), (4, 8, 4, 8))
    )
    centers_um = np.asarray(((22.5, -22.5), (30.0, -30.0)))
    cone_positions_um = np.asarray(
        ((11.25, -11.25), (41.25, -11.25), (11.25, -41.25), (41.25, -41.25))
    )

    # When: canonical BC/AC support is derived from each real contour and extent.
    geometry = build_rf_pathway_geometry(
        masks,
        centers_um,
        np.asarray((30.0, 30.0)),
        ("midget", "parasol"),
        cone_blocks,
        cone_positions_um,
    )

    # Then: AC fills the measured RF support and includes the unchanged BC core.
    assert geometry.spatial_basis.shape == (2, 2, 4)
    assert bool(geometry.bc_support.any(dim=1).all())
    assert bool(geometry.ac_support.any(dim=1).all())
    assert torch.equal(geometry.bc_support * geometry.ac_support, geometry.bc_support)
    assert bool((geometry.ac_support > geometry.bc_support).any(dim=1).all())
    assert bool(geometry.ac_support.all())
    assert bool(((geometry.bc_support + geometry.ac_support) > 0).all(dim=1).all())


def test_explicit_rf_geometry_and_graph_drive_canonical_model() -> None:
    # Given: RF-derived pathway bases and a two-cell non-self locality graph.
    spatial_basis = torch.tensor(
        [[[0.8, 0.2, 0.0], [0.6, 0.3, 0.1]], [[0.0, 0.2, 0.8], [0.1, 0.3, 0.6]]]
    )
    bc_support = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    ac_support = torch.ones(2, 3)
    from models.mechanistic_retina.bipolar_subunits import PathwaySpatialGeometry

    pathway_geometry = PathwaySpatialGeometry(
        spatial_basis=spatial_basis,
        bc_support=bc_support,
        ac_support=ac_support,
    )
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])

    # When: the canonical model is built without radius-derived BC/AC/locality.
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            h1_radius_deg=2.0,
        ),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        ("midget", "midget"),
        ("ON", "ON"),
        shared_subunit_edge_index=edge_index,
        pathway_spatial_geometry=pathway_geometry,
    )

    # Then: buffers and trainable mixing support exactly match the RF contract.
    assert torch.equal(model.feature_bank.spatial_basis, spatial_basis)
    assert torch.equal(model.feature_bank.bc_support, bc_support)
    assert torch.equal(model.feature_bank.ac_support, ac_support)
    assert torch.equal(model.shared_subunits.edge_index, edge_index)


def test_early_stopping_audits_nonself_connection_gradient_and_update() -> None:
    # Given: a small canonical model with asymmetric RF bases and non-self edges.
    from models.mechanistic_retina.bipolar_subunits import PathwaySpatialGeometry

    torch.manual_seed(7)
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    geometry = PathwaySpatialGeometry(
        spatial_basis=torch.tensor(
            [[[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]], [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]
        ),
        bc_support=torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        ac_support=torch.ones(2, 3),
    )
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            lag_steps=3,
            h1_radius_deg=2.0,
        ),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        ("midget", "midget"),
        ("ON", "ON"),
        shared_subunit_edge_index=edge_index,
        pathway_spatial_geometry=geometry,
    )
    cones = torch.randn(6, 8, 3)
    events = torch.bernoulli(torch.full((6, 8, 2), 0.2))

    # When: a bounded early-stopping training run updates the canonical optimizer.
    result = fit_real_spike_model_early_stopping(
        EarlyStoppingTrainingRequest(
            model=model,
            train=_split(cones[:4], events[:4]),
            validation=_split(cones[4:], events[4:]),
            learning_rate=0.01,
            batch_size=2,
            seed=13,
            stopping=EarlyStoppingConfig(
                max_steps=4, evaluation_interval=1, patience=2, min_delta=0.0
            ),
        )
    )

    # Then: non-self connections receive finite signal and change from fresh init.
    assert result.nonself_connection_gradient_nonzero
    assert result.nonself_connection_optimizer_updated
    assert result.nonself_connection_updated
    assert result.nonself_connection_max_abs_gradient > 0.0
    assert result.nonself_connection_update_norm > 0.0
    assert 1 <= result.best_step <= 4
    assert _validation_metrics(model, _split(cones[4:], events[4:])).population_nll == pytest.approx(
        result.best_metrics.population_nll
    )


@pytest.mark.skipif(
    not (_SESSION.exists() and _GRAPH.exists()), reason="Karamanlis artifacts unavailable"
)
def test_rf_population_adapter_uses_sixty_cells_and_no_electrode_proxy() -> None:
    # Given: the measured 60-cell RF artifact and a two-plus-two image split.
    geometry = load_rf_population_geometry(_GRAPH, grid_size=51)
    config = RFPopulationAdapterConfig(
        train_image_count=2,
        validation_image_count=2,
        cell_selection=CellSelection.ALL_QUALITY_1_TARGETS,
    )

    # When: real stimuli and spikes are aligned through RF geometry.
    data = load_rf_population_imagesequence(_SESSION, geometry, config)

    # Then: graph order, real micrometers, and unchanged 51-square capacity agree.
    assert len(data.cell_ids) == 60
    assert data.cell_ids == geometry.cell_ids
    assert data.cone_drive_coordinate_unit == "retinal_micrometers"
    assert data.train.cone_drive.shape[-1] == 51 * 51
    assert data.edge_index.shape == (2, 268)
    assert Counter(zip(data.polarities, data.cell_types, strict=True)) == {
        ("ON", "parasol"): 27,
        ("ON", "midget"): 13,
        ("OFF", "parasol"): 15,
        ("OFF", "midget"): 5,
    }
    assert "electrode" not in data.geometry_lineage.lower()
    blocks = data.cone_blocks_screen_indices.numpy()
    sampled_support = np.asarray(
        [
            [mask[y0:y1, x0:x1].any() for y0, y1, x0, x1 in blocks]
            for mask in geometry.support_masks
        ]
    )
    declared_support = (
        data.pathway_spatial_geometry.bc_support
        + data.pathway_spatial_geometry.ac_support
    ).bool().numpy()
    np.testing.assert_array_equal(sampled_support, declared_support)
    mean_x_px = (blocks[:, 2] + 1 + blocks[:, 3]) / 2
    mean_y_px = (blocks[:, 0] + 1 + blocks[:, 1]) / 2
    expected_um = np.column_stack(
        (
            (mean_x_px - geometry.screen_origin_px[0])
            * geometry.screen_pixel_size_um,
            (geometry.screen_origin_px[1] - mean_y_px)
            * geometry.screen_pixel_size_um,
        )
    )
    np.testing.assert_allclose(data.cone_positions_um.numpy(), expected_um)
    with np.load(geometry.graph_path) as graph_arrays:
        stored_centers = np.asarray(graph_arrays["centers_um"])
    np.testing.assert_allclose(data.cell_positions_um[:, 0], stored_centers[:, 0])
    np.testing.assert_allclose(data.cell_positions_um[:, 1], -stored_centers[:, 1])
    covered = np.zeros_like(geometry.support_masks[0], dtype=bool)
    for y0, y1, x0, x1 in blocks:
        covered[y0:y1, x0:x1] = True
    assert not bool((geometry.support_masks.any(axis=0) & ~covered).any())


@pytest.mark.skipif(not _GRAPH.exists(), reason="Karamanlis graph unavailable")
def test_rf_population_geometry_rejects_wrong_graph_fingerprint() -> None:
    # Given: the canonical graph directory with an incorrect expected fingerprint.
    invalid_sha256 = "0" * 64

    # When: training geometry is loaded against that immutable lineage claim.
    from data.karamanlis_rf_population import RFPopulationDataError

    # Then: the artifact is rejected before its arrays can enter the model.
    with pytest.raises(RFPopulationDataError, match="fingerprint"):
        load_rf_population_geometry(
            _GRAPH,
            grid_size=51,
            expected_graph_sha256=invalid_sha256,
        )
