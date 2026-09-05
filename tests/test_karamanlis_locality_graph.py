from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluation.mechanistic_retina as retina_evaluation
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.shared_subunits import (
    SharedSubunitLayout,
    SharedSubunitMixer,
)


_RF_SOURCE = Path("output/real_data/karamanlis_2024_population_rf_centers_v1")


def _locality_api():
    # Given: the mechanistic-retina evaluation package.
    names = (
        "RFMapGrid",
        "RFSpatialExtent",
        "RFLocalityCell",
        "build_rf_locality_graph",
        "extract_rf_spatial_extent",
        "run_karamanlis_locality_graph",
    )

    # When: the locality-graph API is resolved.
    values = tuple(getattr(retina_evaluation, name, None) for name in names)

    # Then: every typed boundary required by the graph contract exists.
    assert all(callable(value) for value in values)
    return values


def _extent(extent_type, center_um: tuple[float, float], support: np.ndarray):
    area_um2 = float(support.sum())
    contour_um = np.argwhere(support)[:, ::-1].astype(np.float64)
    return extent_type(
        support_mask=support,
        contour_um=contour_um,
        center_um=np.asarray(center_um, dtype=np.float64),
        area_um2=area_um2,
        equivalent_radius_um=float(np.sqrt(area_um2 / np.pi)),
        width_um=float(np.ptp(contour_um[:, 0]) + 1.0),
        height_um=float(np.ptp(contour_um[:, 1]) + 1.0),
    )


def test_rf_extent_uses_experimental_micrometers_and_twenty_five_percent_contour() -> None:
    # Given: a synthetic RF map on an experimentally projected 7.5-um pixel grid.
    grid_type, _, _, _, extract_extent, _ = _locality_api()
    y, x = np.mgrid[0:17, 0:19]
    spatial_rf = np.exp(-((x - 12.0) ** 2 + (y - 6.0) ** 2) / 8.0)
    grid = grid_type(
        x_centers_px=np.arange(19, dtype=np.float64),
        y_centers_px=np.arange(17, dtype=np.float64),
        stixel_width_px=1,
        stixel_height_px=1,
        screen_pixel_size_um=7.5,
        origin_px=np.asarray([9.0, 8.0]),
    )

    # When: the central 25%-of-peak component is converted to retinal geometry.
    extent = extract_extent(spatial_rf, grid)

    # Then: its center and size are expressed directly in retinal micrometers.
    np.testing.assert_allclose(extent.center_um, (22.5, -15.0), atol=4.0)
    assert extent.area_um2 > 0.0
    assert extent.equivalent_radius_um > 0.0
    assert extent.contour_um.shape[1] == 2


def test_locality_graph_uses_overlap_or_extent_without_cross_class_edges() -> None:
    # Given: same-class near/far cells and overlapping cells from other classes.
    _, extent_type, cell_type, build_graph, _, _ = _locality_api()
    support_a = np.zeros((8, 8), dtype=bool)
    support_a[1:3, 1:3] = True
    support_b = np.zeros((8, 8), dtype=bool)
    support_b[1:3, 3:5] = True
    support_far = np.zeros((8, 8), dtype=bool)
    support_far[5:7, 5:7] = True
    cells = (
        cell_type(0, "a", "parasol", "ON", _extent(extent_type, (1.5, 1.5), support_a)),
        cell_type(1, "b", "parasol", "ON", _extent(extent_type, (3.5, 1.5), support_b)),
        cell_type(2, "c", "parasol", "OFF", _extent(extent_type, (1.5, 1.5), support_a)),
        cell_type(3, "d", "midget", "ON", _extent(extent_type, (1.5, 1.5), support_a)),
        cell_type(4, "e", "parasol", "ON", _extent(extent_type, (5.5, 5.5), support_far)),
    )

    # When: locality is built from RF overlap or extent-normalized proximity.
    graph = build_graph(cells)

    # Then: a/b connect, every self-edge remains, and cross-class/far pairs do not.
    assert bool(graph.adjacency[0, 1] and graph.adjacency[1, 0])
    assert bool(np.diag(graph.adjacency).all())
    assert not bool(graph.adjacency[0, 2])
    assert not bool(graph.adjacency[0, 3])
    assert not bool(graph.adjacency[0, 4])


def test_raw_connections_match_canonical_positive_initialization() -> None:
    # Given: a two-cell same-class graph connected by RF extent.
    _, extent_type, cell_type, build_graph, _, _ = _locality_api()
    support = np.ones((2, 2), dtype=bool)
    cells = (
        cell_type(0, "a", "midget", "ON", _extent(extent_type, (0.0, 0.0), support)),
        cell_type(1, "b", "midget", "ON", _extent(extent_type, (1.0, 0.0), support)),
    )

    # When: raw_connections are generated in edge-index order.
    graph = build_graph(cells)
    positive = np.logaddexp(0.0, graph.raw_connections)

    # Then: self edges initialize to 1.0 and non-self edges to 0.1.
    expected = np.where(graph.edge_index[0] == graph.edge_index[1], 1.0, 0.1)
    np.testing.assert_allclose(positive, expected, atol=1e-7)


def test_explicit_edges_control_shared_subunit_support() -> None:
    # Given: three cells whose explicit RF graph connects only cells zero and one.
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
    layout = SharedSubunitLayout(
        cell_positions=torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        cell_types=("midget", "midget", "midget"),
        polarities=("ON", "ON", "ON"),
        edge_index=edge_index,
    )

    # When: the canonical mixer is initialized with the explicit support.
    mixer = SharedSubunitMixer(layout, radius_deg=0.01, trainable=True)

    # Then: the supplied target/source edges override the radius support exactly.
    assert torch.equal(mixer.edge_index, edge_index)
    expected_support = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, True]]
    )
    assert torch.equal(mixer.connection_matrix() > 0, expected_support)


def test_canonical_model_accepts_explicit_shared_subunit_graph() -> None:
    # Given: an RF-derived support that differs from radius-based locality.
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
    cone_positions = torch.tensor(
        [[0.0, 0.0], [0.04, 0.0], [0.08, 0.0], [0.12, 0.0], [0.16, 0.0]]
    )
    cell_positions = torch.tensor([[0.0, 0.0], [0.04, 0.0], [0.12, 0.0]])

    # When: the canonical model is built with that explicit graph.
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(architecture_mode="mechanism_identifiable"),
        cone_positions,
        cell_positions,
        ("midget", "midget", "midget"),
        ("ON", "ON", "ON"),
        shared_subunit_edge_index=edge_index,
    )

    # Then: canonical forward support preserves the supplied target/source edges.
    assert torch.equal(model.shared_subunits.edge_index, edge_index)


@pytest.mark.skipif(not _RF_SOURCE.exists(), reason="RF-center artifact unavailable")
def test_real_rf_artifact_builds_sixty_cell_micrometer_graph(tmp_path: Path) -> None:
    # Given: the independently extracted 64-cell RF artifact with split-half QC.
    *_, run_graph = _locality_api()

    # When: only reliable cells are converted into the training-ready graph artifact.
    result = run_graph(_RF_SOURCE, tmp_path / "graph")
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    arrays = np.load(result.graph_path)

    # Then: four unstable cells are absent and the graph uses only retinal micrometers.
    assert payload["summary"]["cell_count"] == 60
    assert payload["coordinates"]["unit"] == "retinal_micrometers"
    assert payload["coordinates"]["y_positive"] == "down"
    assert "degree" not in json.dumps(payload).lower()
    assert set(payload["excluded_cell_ids"]) == {"67", "3367", "3908", "19249"}
    assert int(arrays["adjacency"].sum()) > 60
    assert arrays["edge_index"].shape[0] == 2
