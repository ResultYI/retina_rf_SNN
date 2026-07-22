from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.two_cluster import ClusterResult
from models.cells.rgc_types import RGCOutput


@dataclass(frozen=True, slots=True)
class RGCTypeReport:
    trained_features: np.ndarray
    initialized_features: np.ndarray
    standardized_trained_features: np.ndarray
    standardized_initialized_features: np.ndarray
    eligibility_mask: np.ndarray
    initialized_valid_response_mask: np.ndarray
    assignments: np.ndarray
    initialized_assignments: np.ndarray
    cluster_names: tuple[str, str]
    candidate_labels: tuple[str | None, str | None]
    status: str
    trained_silhouette: float
    initialized_silhouette: float
    trained_minimum_cluster_fraction: float
    initialized_minimum_cluster_fraction: float
    relative_radius_difference: float
    sustained_difference: float
    flicker_difference: float
    initialized_between_cluster_separation: float
    trained_between_cluster_separation: float
    trained_to_initial_separation_ratio: float | None
    absolute_separation_gain: float
    hard_active_fraction_by_polarity_unit: np.ndarray


def build_rgc_type_report(
    trained_features: np.ndarray,
    initialized_features: np.ndarray,
    standardized_trained: np.ndarray,
    standardized_initialized: np.ndarray,
    eligibility: np.ndarray,
    initialized_valid_response_mask: np.ndarray,
    assignments: np.ndarray,
    initialized_assignments: np.ndarray,
    output: RGCOutput,
    *,
    status: str,
    labels: tuple[str | None, str | None] = (None, None),
    trained_cluster: ClusterResult | None = None,
    initialized_cluster: ClusterResult | None = None,
    relative_radius_difference: float = 0.0,
    sustained_difference: float = 0.0,
    flicker_difference: float = 0.0,
    ratio: float | None = None,
    absolute_gain: float = 0.0,
) -> RGCTypeReport:
    return RGCTypeReport(
        trained_features=trained_features,
        initialized_features=initialized_features,
        standardized_trained_features=standardized_trained,
        standardized_initialized_features=standardized_initialized,
        eligibility_mask=eligibility,
        initialized_valid_response_mask=initialized_valid_response_mask,
        assignments=assignments,
        initialized_assignments=initialized_assignments,
        cluster_names=("cluster 0", "cluster 1"),
        candidate_labels=labels,
        status=status,
        trained_silhouette=trained_cluster.silhouette if trained_cluster else 0.0,
        initialized_silhouette=(
            initialized_cluster.silhouette if initialized_cluster else 0.0
        ),
        trained_minimum_cluster_fraction=(
            trained_cluster.minimum_fraction if trained_cluster else 0.0
        ),
        initialized_minimum_cluster_fraction=(
            initialized_cluster.minimum_fraction if initialized_cluster else 0.0
        ),
        relative_radius_difference=relative_radius_difference,
        sustained_difference=sustained_difference,
        flicker_difference=flicker_difference,
        initialized_between_cluster_separation=(
            initialized_cluster.separation if initialized_cluster else 0.0
        ),
        trained_between_cluster_separation=(
            trained_cluster.separation if trained_cluster else 0.0
        ),
        trained_to_initial_separation_ratio=ratio,
        absolute_separation_gain=absolute_gain,
        hard_active_fraction_by_polarity_unit=(
            (output.hard_spikes > 0).float().mean(dim=(0, 1)).cpu().numpy()
        ),
    )


__all__ = ["RGCTypeReport", "build_rgc_type_report"]
