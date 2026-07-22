from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_types import RGCOutput
from evaluation.temporal_probes import TemporalProbeFeatures


class RGCTypeError(ValueError):
    pass


FEATURE_NAMES = (
    "effective_spatial_radius",
    "sustained_mix",
    "impulse_peak",
    "impulse_time_to_peak_ms",
    "impulse_width_ms",
    "step_sustained_index",
    "flicker_response",
    "membrane_tau",
    "adaptation_tau",
    "adaptation_gain",
    "amacrine_gain",
    "mean_rate",
    "active_fraction",
)


@dataclass(frozen=True, slots=True)
class RGCTypeReport:
    features: np.ndarray
    standardized_features: np.ndarray
    assignments: np.ndarray
    cluster_names: tuple[str, str]
    candidate_labels: tuple[str | None, str | None]
    status: str
    silhouette: float
    minimum_cluster_fraction: float
    relative_radius_difference: float
    sustained_difference: float
    initial_phenotype_separation: float
    current_phenotype_separation: float
    hard_active_fraction_by_polarity_unit: np.ndarray


def identify_rgc_types(
    rgc: HeterogeneousRGCPool,
    output: RGCOutput,
    *,
    probes: TemporalProbeFeatures,
    seed: int = 0,
) -> RGCTypeReport:
    unit_count = rgc.unit_count
    if output.rates.ndim != 4 or output.rates.shape[-1] != unit_count:
        raise RGCTypeError("RGC output must have shape [batch,time,polarity,unit]")
    weights = rgc.compute_spatial_weights().detach()
    effective_radius = torch.sqrt(
        (weights * rgc.distance_sq_degs).sum(dim=1).clamp_min(0.0)
    )
    mean_rate = output.rates.mean(dim=(0, 1, 2))
    active_fraction = (output.hard_spikes > 0).float().mean(dim=(0, 1, 2))
    features = torch.stack(
        (
            effective_radius,
            rgc.sustained_mix.detach(),
            probes.impulse_peak.detach(),
            probes.impulse_time_to_peak_ms.detach(),
            probes.impulse_width_ms.detach(),
            probes.step_sustained_index.detach(),
            probes.flicker_response.detach(),
            rgc.membrane_tau_ms.detach(),
            rgc.adaptation_tau_ms.detach(),
            rgc.adaptation_gain.detach(),
            rgc.amacrine_gain.detach(),
            mean_rate.detach(),
            active_fraction.detach(),
        ),
        dim=1,
    ).cpu().numpy()
    standardized = _standardize(features)
    assignments = _kmeans_two(standardized, seed)
    fractions = np.bincount(assignments, minlength=2) / assignments.size
    minimum_fraction = float(fractions.min())
    if minimum_fraction == 0.0:
        silhouette = 0.0
        radius_difference = 0.0
        sustained_difference = 0.0
    else:
        silhouette = _silhouette(standardized, assignments)
        means = np.stack(
            [features[assignments == cluster].mean(axis=0) for cluster in range(2)]
        )
        radius_difference = float(
            abs(means[0, 0] - means[1, 0])
            / max(abs(features[:, 0].mean()), 1e-8)
        )
        sustained_difference = float(abs(means[0, 5] - means[1, 5]))
    stable = (
        minimum_fraction >= 0.20
        and silhouette >= 0.25
        and radius_difference >= 0.10
        and sustained_difference >= 0.10
    )
    labels = _candidate_labels(features, assignments) if stable else (None, None)
    return RGCTypeReport(
        features=features,
        standardized_features=standardized,
        assignments=assignments,
        cluster_names=("cluster 0", "cluster 1"),
        candidate_labels=labels,
        status=(
            "stable two-cluster structure"
            if stable
            else "no stable two-cluster structure"
        ),
        silhouette=silhouette,
        minimum_cluster_fraction=minimum_fraction,
        relative_radius_difference=radius_difference,
        sustained_difference=sustained_difference,
        initial_phenotype_separation=_phenotype_separation(
            rgc.initial_phenotype_features, rgc.unit_center_indices
        ),
        current_phenotype_separation=_phenotype_separation(
            rgc.phenotype_features().detach(), rgc.unit_center_indices
        ),
        hard_active_fraction_by_polarity_unit=(
            (output.hard_spikes > 0).float().mean(dim=(0, 1)).cpu().numpy()
        ),
    )


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(scale, 1e-8)


def _kmeans_two(features: np.ndarray, seed: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 2:
        raise RGCTypeError("k-means requires at least two units")
    generator = np.random.default_rng(seed)
    first = int(generator.integers(features.shape[0]))
    distances = np.square(features - features[first]).sum(axis=1)
    second = int(np.argmax(distances))
    centers = features[[first, second]].copy()
    assignments = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(100):
        next_assignments = np.square(
            features[:, None, :] - centers[None, :, :]
        ).sum(axis=2).argmin(axis=1)
        if np.array_equal(next_assignments, assignments) and _ > 0:
            break
        assignments = next_assignments
        for cluster in range(2):
            members = features[assignments == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    return assignments


def _silhouette(features: np.ndarray, assignments: np.ndarray) -> float:
    values: list[float] = []
    for index, row in enumerate(features):
        own = assignments == assignments[index]
        own[index] = False
        other = assignments != assignments[index]
        if not own.any() or not other.any():
            values.append(0.0)
            continue
        within = np.linalg.norm(features[own] - row, axis=1).mean()
        between = np.linalg.norm(features[other] - row, axis=1).mean()
        values.append(float((between - within) / max(within, between, 1e-8)))
    return float(np.mean(values))


def _phenotype_separation(
    phenotype: torch.Tensor,
    center_indices: torch.Tensor,
) -> float:
    distances: list[torch.Tensor] = []
    for center in torch.unique(center_indices):
        members = phenotype[center_indices == center]
        if members.shape[0] < 2:
            continue
        pairwise = torch.pdist(members)
        if pairwise.numel():
            distances.append(pairwise)
    if not distances:
        return 0.0
    return float(torch.cat(distances).mean())


def _candidate_labels(
    features: np.ndarray,
    assignments: np.ndarray,
) -> tuple[str | None, str | None]:
    if any(not np.any(assignments == cluster) for cluster in range(2)):
        return None, None
    means = np.stack(
        [features[assignments == cluster].mean(axis=0) for cluster in range(2)]
    )
    smaller = int(np.argmin(means[:, 0]))
    larger = 1 - smaller
    preregistered = (
        means[smaller, 0] < means[larger, 0]
        and means[smaller, 1] > means[larger, 1]
        and means[smaller, 5] > means[larger, 5]
        and means[larger, 2] > means[smaller, 2]
    )
    if not preregistered:
        return None, None
    labels: list[str | None] = [None, None]
    labels[smaller] = "midget-like candidate"
    labels[larger] = "parasol-like candidate"
    return labels[0], labels[1]


__all__ = [
    "FEATURE_NAMES",
    "RGCTypeError",
    "RGCTypeReport",
    "identify_rgc_types",
]
