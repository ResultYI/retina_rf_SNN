from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.temporal_probes import TemporalProbeFeatures
from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_types import RGCOutput
from training.config import EvaluationConfig


class RGCTypeError(ValueError):
    pass


FEATURE_NAMES = (
    "effective_spatial_radius",
    "impulse_time_to_peak_ms",
    "impulse_width_ms",
    "step_sustained_index",
    "normalized_flicker_response",
)


@dataclass(frozen=True, slots=True)
class RGCTypeReport:
    trained_features: np.ndarray
    initialized_features: np.ndarray
    standardized_trained_features: np.ndarray
    standardized_initialized_features: np.ndarray
    eligibility_mask: np.ndarray
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


@dataclass(frozen=True, slots=True)
class _ClusterResult:
    assignments: np.ndarray
    silhouette: float
    minimum_fraction: float
    means: np.ndarray
    separation: float


def identify_rgc_types(
    trained_rgc: HeterogeneousRGCPool,
    trained_output: RGCOutput,
    *,
    probes: TemporalProbeFeatures,
    initialized_rgc: HeterogeneousRGCPool,
    initialized_probes: TemporalProbeFeatures,
    config: EvaluationConfig,
    seed: int = 0,
) -> RGCTypeReport:
    unit_count = trained_rgc.unit_count
    if initialized_rgc.unit_count != unit_count:
        raise RGCTypeError("Initialized and trained RGC unit counts must match")
    if trained_output.rates.ndim != 4 or trained_output.rates.shape[-1] != unit_count:
        raise RGCTypeError("RGC output must have shape [batch,time,polarity,unit]")
    trained_features = _functional_features(trained_rgc, probes)
    initialized_features = _functional_features(initialized_rgc, initialized_probes)
    eligibility = (
        _probe_eligibility(probes)
        & _probe_eligibility(initialized_probes)
        & np.isfinite(trained_features).all(axis=1)
        & np.isfinite(initialized_features).all(axis=1)
    )
    assignments = np.full(unit_count, -1, dtype=np.int64)
    initialized_assignments = np.full(unit_count, -1, dtype=np.int64)
    eligible_count = int(eligibility.sum())
    if eligible_count < 2:
        return _report(
            trained_features,
            initialized_features,
            trained_features.copy(),
            initialized_features.copy(),
            eligibility,
            assignments,
            initialized_assignments,
            trained_output,
            status="not_identifiable",
        )

    stacked = np.concatenate(
        (trained_features[eligibility], initialized_features[eligibility]),
        axis=0,
    )
    mean = stacked.mean(axis=0, keepdims=True)
    scale = np.maximum(stacked.std(axis=0, keepdims=True), 1e-8)
    standardized_trained = (trained_features - mean) / scale
    standardized_initialized = (initialized_features - mean) / scale
    trained_cluster = _cluster_two(standardized_trained[eligibility], seed)
    initialized_cluster = _cluster_two(standardized_initialized[eligibility], seed)
    assignments[eligibility] = trained_cluster.assignments
    initialized_assignments[eligibility] = initialized_cluster.assignments
    trained_means = np.stack(
        [
            trained_features[assignments == cluster].mean(axis=0)
            for cluster in range(2)
        ]
    ) if trained_cluster.minimum_fraction > 0.0 else np.zeros((2, len(FEATURE_NAMES)))
    relative_radius_difference = _relative_difference(trained_means[:, 0])
    sustained_difference = float(abs(np.diff(trained_means[:, 3])[0]))
    flicker_difference = float(abs(np.diff(trained_means[:, 4])[0]))
    stable = (
        trained_cluster.minimum_fraction >= config.rgc_min_cluster_fraction
        and trained_cluster.silhouette >= config.rgc_min_silhouette
        and relative_radius_difference >= config.rgc_min_relative_radius_difference
        and sustained_difference >= config.rgc_min_sustained_difference
        and flicker_difference >= config.rgc_min_flicker_difference
    )
    initial_separation = initialized_cluster.separation
    trained_separation = trained_cluster.separation
    absolute_gain = trained_separation - initial_separation
    ratio = (
        trained_separation / initial_separation
        if initial_separation > 1e-8
        else None
    )
    labels = (
        _candidate_labels(trained_features, assignments, config)
        if stable
        else (None, None)
    )
    pairing = all(label is not None for label in labels)
    learned = stable and pairing and (
        (
            ratio is not None
            and ratio >= config.rgc_min_trained_to_initial_separation_ratio
        )
        or (
            initial_separation < config.rgc_min_absolute_separation_gain
            and absolute_gain >= config.rgc_min_absolute_separation_gain
        )
    )
    if learned:
        status = "learned_functional_pairing_candidate"
    elif stable and pairing:
        status = "initialization_level_or_architecture_induced_separation"
    elif not stable and trained_cluster.silhouette > 0.0 and trained_cluster.separation > 0.0:
        status = "continuous_heterogeneity_without_stable_clusters"
    else:
        status = "no_stable_two_cluster_structure"
    return _report(
        trained_features,
        initialized_features,
        standardized_trained,
        standardized_initialized,
        eligibility,
        assignments,
        initialized_assignments,
        trained_output,
        status=status,
        labels=labels,
        trained_cluster=trained_cluster,
        initialized_cluster=initialized_cluster,
        relative_radius_difference=relative_radius_difference,
        sustained_difference=sustained_difference,
        flicker_difference=flicker_difference,
        ratio=ratio,
        absolute_gain=absolute_gain,
    )


def _functional_features(
    rgc: HeterogeneousRGCPool,
    probes: TemporalProbeFeatures,
) -> np.ndarray:
    weights = rgc.compute_spatial_weights().detach()
    effective_radius = torch.sqrt(
        (weights * rgc.distance_sq_degs).sum(dim=1).clamp_min(0.0)
    )
    tensors = (
        effective_radius,
        probes.impulse_time_to_peak_ms,
        probes.impulse_width_ms,
        probes.step_sustained_index,
        probes.flicker_response,
    )
    if any(tensor.shape != (rgc.unit_count,) for tensor in tensors):
        raise RGCTypeError("Every functional feature must have shape [unit]")
    return torch.stack(tuple(tensor.detach() for tensor in tensors), dim=1).cpu().numpy()


def _probe_eligibility(probes: TemporalProbeFeatures) -> np.ndarray:
    continuous_quality = probes.impulse_peak >= 1e-4
    hard_quality = probes.hard_evoked_spike_count > 0
    return (
        probes.valid_response_mask
        & (continuous_quality | hard_quality)
    ).detach().cpu().numpy().astype(bool)


def _cluster_two(features: np.ndarray, seed: int) -> _ClusterResult:
    assignments = _kmeans_two(features, seed)
    fractions = np.bincount(assignments, minlength=2) / assignments.size
    if fractions.min() == 0.0:
        return _ClusterResult(assignments, 0.0, 0.0, np.zeros((2, features.shape[1])), 0.0)
    means = np.stack(
        [features[assignments == cluster].mean(axis=0) for cluster in range(2)]
    )
    return _ClusterResult(
        assignments=assignments,
        silhouette=_silhouette(features, assignments),
        minimum_fraction=float(fractions.min()),
        means=means,
        separation=float(np.linalg.norm(means[0] - means[1])),
    )


def _kmeans_two(features: np.ndarray, seed: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 2:
        raise RGCTypeError("k-means requires at least two eligible units")
    generator = np.random.default_rng(seed)
    first = int(generator.integers(features.shape[0]))
    distances = np.square(features - features[first]).sum(axis=1)
    centers = features[[first, int(np.argmax(distances))]].copy()
    assignments = np.zeros(features.shape[0], dtype=np.int64)
    for iteration in range(100):
        next_assignments = np.square(
            features[:, None, :] - centers[None, :, :]
        ).sum(axis=2).argmin(axis=1)
        if iteration > 0 and np.array_equal(next_assignments, assignments):
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


def _candidate_labels(
    features: np.ndarray,
    assignments: np.ndarray,
    config: EvaluationConfig,
) -> tuple[str | None, str | None]:
    means = np.stack(
        [features[assignments == cluster].mean(axis=0) for cluster in range(2)]
    )
    smaller = int(np.argmin(means[:, 0]))
    larger = 1 - smaller
    time_tolerance = 0.10 * max(float(np.ptp(features[:, 1])), 1e-8)
    pairing = (
        means[smaller, 3] - means[larger, 3] >= config.rgc_min_sustained_difference
        and means[larger, 4] - means[smaller, 4] >= config.rgc_min_flicker_difference
        and means[larger, 1] <= means[smaller, 1] + time_tolerance
    )
    if not pairing:
        return None, None
    labels: list[str | None] = [None, None]
    labels[smaller] = "midget-like candidate"
    labels[larger] = "parasol-like candidate"
    return labels[0], labels[1]


def _relative_difference(values: np.ndarray) -> float:
    return float(abs(values[0] - values[1]) / max(abs(values.mean()), 1e-8))


def _report(
    trained_features: np.ndarray,
    initialized_features: np.ndarray,
    standardized_trained: np.ndarray,
    standardized_initialized: np.ndarray,
    eligibility: np.ndarray,
    assignments: np.ndarray,
    initialized_assignments: np.ndarray,
    output: RGCOutput,
    *,
    status: str,
    labels: tuple[str | None, str | None] = (None, None),
    trained_cluster: _ClusterResult | None = None,
    initialized_cluster: _ClusterResult | None = None,
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


__all__ = [
    "FEATURE_NAMES",
    "RGCTypeError",
    "RGCTypeReport",
    "identify_rgc_types",
]
