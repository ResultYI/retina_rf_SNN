from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import torch

from models.mechanistic_retina.pathway_spatial_geometry import PathwaySpatialGeometry


_GRAPH_SCHEMA: Final = "karamanlis_marmoset_rf_locality_graph_v1"
_EXPECTED_CELL_COUNT: Final = 60
_EXPECTED_EDGE_COUNT: Final = 268
_BC_EXTENT_FRACTION: Final = {"midget": 0.06 / 0.13, "parasol": 0.10 / 0.15}
_SPATIAL_SIGMA_FRACTIONS: Final = {
    "midget": (0.05 / 0.13, 0.14 / 0.13),
    "parasol": (0.09 / 0.15, 0.20 / 0.15),
}


@dataclass(frozen=True, slots=True)
class RFPopulationGeometry:
    session_id: str
    cell_ids: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    centers_um: np.ndarray
    equivalent_radii_um: np.ndarray
    support_masks: np.ndarray
    edge_index: torch.Tensor
    screen_origin_px: np.ndarray
    screen_pixel_size_um: float
    crop_center_px: np.ndarray
    crop_pixels: int
    pool_factor: int
    grid_size: int
    graph_path: Path


class RFPopulationDataError(ValueError):
    pass


def load_rf_population_geometry(
    artifact_dir: str | Path,
    *,
    grid_size: int,
    expected_graph_sha256: str | None = None,
) -> RFPopulationGeometry:
    source = Path(artifact_dir)
    graph_path = source / "locality_graph.npz"
    if expected_graph_sha256 is not None:
        actual_graph_sha256 = _sha256(graph_path)
        if actual_graph_sha256 != expected_graph_sha256.lower():
            raise RFPopulationDataError("RF locality graph fingerprint differs")
    payload = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if payload.get("schema") != _GRAPH_SCHEMA or grid_size < 1:
        raise RFPopulationDataError("RF locality artifact contract is unsupported")
    with np.load(graph_path) as arrays:
        ids = tuple(str(value) for value in arrays["cell_ids"].tolist())
        types = tuple(str(value) for value in arrays["cell_types"].tolist())
        polarities = tuple(str(value) for value in arrays["polarities"].tolist())
        centers = np.asarray(arrays["centers_um"], dtype=np.float64)
        centers[:, 1] *= -1.0
        radii = np.asarray(arrays["equivalent_radii_um"], dtype=np.float64)
        masks = np.asarray(arrays["support_masks"], dtype=bool)
        edges = np.asarray(arrays["edge_index"], dtype=np.int64)
    if len(ids) != _EXPECTED_CELL_COUNT or edges.shape != (2, _EXPECTED_EDGE_COUNT):
        raise RFPopulationDataError("RF locality artifact dimensions differ")
    union_y, union_x = np.nonzero(masks.any(axis=0))
    required = max(int(np.ptp(union_x)) + 1, int(np.ptp(union_y)) + 1)
    pool_factor = int(np.ceil(required / grid_size))
    crop_pixels = pool_factor * grid_size
    if crop_pixels % 2 == 0:
        raise RFPopulationDataError("RF-derived crop must remain odd")
    center = np.asarray(
        (
            round((union_x.min() + union_x.max() + 2) / 2),
            round((union_y.min() + union_y.max() + 2) / 2),
        ),
        dtype=np.float64,
    )
    coordinates = payload["coordinates"]
    if coordinates.get("y_positive") != "down":
        raise RFPopulationDataError(
            "RF locality source must declare its screen-y-down array encoding"
        )
    return RFPopulationGeometry(
        session_id=str(payload["session_id"]),
        cell_ids=ids,
        cell_types=types,
        polarities=polarities,
        centers_um=centers,
        equivalent_radii_um=radii,
        support_masks=masks,
        edge_index=torch.from_numpy(edges.copy()),
        screen_origin_px=np.asarray(
            coordinates["origin_screen_pixels"], dtype=np.float64
        ),
        screen_pixel_size_um=float(coordinates["screen_pixel_size_um"]),
        crop_center_px=center,
        crop_pixels=crop_pixels,
        pool_factor=pool_factor,
        grid_size=grid_size,
        graph_path=graph_path,
    )


def build_rf_pathway_geometry(
    support_masks: np.ndarray,
    centers_um: np.ndarray,
    equivalent_radii_um: np.ndarray,
    cell_types: tuple[str, ...],
    cone_blocks_screen_indices: np.ndarray,
    cone_positions_um: np.ndarray,
) -> PathwaySpatialGeometry:
    sampled = np.asarray(
        [
            [
                mask[y0:y1, x0:x1].any()
                for y0, y1, x0, x1 in cone_blocks_screen_indices
            ]
            for mask in support_masks
        ],
        dtype=bool,
    )
    distances = np.linalg.norm(
        centers_um[:, None] - cone_positions_um[None], axis=-1
    )
    bc = np.zeros_like(sampled)
    basis = np.empty(
        (len(cell_types), 2, cone_positions_um.shape[0]), dtype=np.float32
    )
    for index, cell_type in enumerate(cell_types):
        core_radius = _BC_EXTENT_FRACTION[cell_type] * equivalent_radii_um[index]
        inside_distances = distances[index, sampled[index]]
        if inside_distances.size == 0:
            raise RFPopulationDataError(
                "an RF contour misses every projected cone block"
            )
        core_radius = max(core_radius, float(inside_distances.min()) + 1e-6)
        bc[index] = sampled[index] & (distances[index] <= core_radius)
        for mode, fraction in enumerate(_SPATIAL_SIGMA_FRACTIONS[cell_type]):
            sigma = max(float(equivalent_radii_um[index] * fraction), 1e-6)
            basis[index, mode] = np.exp(-0.5 * (distances[index] / sigma) ** 2)
    ac = sampled
    if not np.all(bc.any(axis=1) & (ac & ~bc).any(axis=1)):
        raise RFPopulationDataError(
            "RF-derived AC support must include and extend beyond nonempty BC support"
        )
    basis /= np.maximum(basis.sum(axis=-1, keepdims=True), 1e-12)
    return PathwaySpatialGeometry(
        spatial_basis=torch.from_numpy(basis),
        bc_support=torch.from_numpy(bc.astype(np.float32)),
        ac_support=torch.from_numpy(ac.astype(np.float32)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "RFPopulationDataError",
    "RFPopulationGeometry",
    "build_rf_pathway_geometry",
    "load_rf_population_geometry",
]
