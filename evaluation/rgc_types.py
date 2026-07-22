from __future__ import annotations

import numpy as np
import torch

from evaluation.rgc_type_report import RGCTypeReport, build_rgc_type_report
from evaluation.temporal_probes import TemporalProbeFeatures
from evaluation.two_cluster import cluster_two
from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_types import RGCOutput
from training.config import EvaluationConfig


class RGCTypeError(ValueError):
    pass


FEATURE_NAMES = (
    "encoder_pooling_radius",
    "impulse_time_to_peak_ms",
    "impulse_width_ms",
    "step_sustained_index",
    "normalized_flicker_response",
)


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
    initialized_valid_response_mask = _probe_eligibility(initialized_probes)
    eligibility = eligible_rgc_units(probes, trained_features, initialized_features)
    assignments = np.full(unit_count, -1, dtype=np.int64)
    initialized_assignments = np.full(unit_count, -1, dtype=np.int64)
    eligible_count = int(eligibility.sum())
    if eligible_count < 2:
        return build_rgc_type_report(
            trained_features,
            initialized_features,
            trained_features.copy(),
            initialized_features.copy(),
            eligibility,
            initialized_valid_response_mask,
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
    trained_cluster = cluster_two(standardized_trained[eligibility], seed)
    initialized_cluster = cluster_two(standardized_initialized[eligibility], seed)
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
    learned = (
        stable
        and pairing
        and separation_supports_learning(
            initial_separation,
            trained_separation,
            config,
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
    return build_rgc_type_report(
        trained_features,
        initialized_features,
        standardized_trained,
        standardized_initialized,
        eligibility,
        initialized_valid_response_mask,
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
    encoder_pooling_radius = torch.sqrt(
        (weights * rgc.distance_sq_degs).sum(dim=1).clamp_min(0.0)
    )
    tensors = (
        encoder_pooling_radius,
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


def eligible_rgc_units(
    trained_probes: TemporalProbeFeatures,
    trained_features: np.ndarray,
    initialized_features: np.ndarray,
) -> np.ndarray:
    return (
        _probe_eligibility(trained_probes)
        & np.isfinite(trained_features).all(axis=1)
        & np.isfinite(initialized_features).all(axis=1)
    )


def separation_supports_learning(
    initialized_separation: float,
    trained_separation: float,
    config: EvaluationConfig,
) -> bool:
    minimum_gain = config.rgc_min_absolute_separation_gain
    absolute_gain = trained_separation - initialized_separation
    if initialized_separation < minimum_gain:
        return absolute_gain >= minimum_gain
    return bool(
        absolute_gain >= minimum_gain
        and trained_separation / initialized_separation
        >= config.rgc_min_trained_to_initial_separation_ratio
    )


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


__all__ = [
    "FEATURE_NAMES",
    "RGCTypeError",
    "RGCTypeReport",
    "eligible_rgc_units",
    "identify_rgc_types",
    "separation_supports_learning",
]
