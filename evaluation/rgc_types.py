from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_types import RGCOutput


class RGCTypeError(ValueError):
    pass


FEATURE_NAMES = (
    "effective_spatial_radius",
    "sustained_mix",
    "impulse_peak",
    "step_sustained_index",
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


def identify_rgc_types(
    rgc: HeterogeneousRGCPool,
    output: RGCOutput,
    *,
    impulse_peak: torch.Tensor | None = None,
    step_sustained_index: torch.Tensor | None = None,
    seed: int = 0,
) -> RGCTypeReport:
    unit_count = rgc.unit_count
    if output.rates.ndim != 4 or output.rates.shape[-1] != unit_count:
        raise RGCTypeError("RGC output must have shape [batch,time,polarity,unit]")
    weights = rgc.compute_spatial_weights().detach()
    effective_radius = torch.sqrt(
        (weights * rgc.distance_sq_degs).sum(dim=1).clamp_min(0.0)
    )
    temporal_trace = output.spike_probability.mean(dim=(0, 2))
    default_peak = temporal_trace.amax(dim=0)
    default_sustained = temporal_trace[-max(1, temporal_trace.shape[0] // 8) :].mean(dim=0) / default_peak.clamp_min(1e-8)
    mean_rate = output.rates.mean(dim=(0, 1, 2))
    active_fraction = (output.hard_spikes > 0).float().mean(dim=(0, 1, 2))
    features = torch.stack(
        (
            effective_radius,
            rgc.sustained_mix.detach(),
            default_peak if impulse_peak is None else impulse_peak.detach(),
            default_sustained if step_sustained_index is None else step_sustained_index.detach(),
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
    labels = _candidate_labels(features, assignments)
    return RGCTypeReport(
        features=features,
        standardized_features=standardized,
        assignments=assignments,
        cluster_names=("cluster 0", "cluster 1"),
        candidate_labels=labels,
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
        and means[smaller, 3] > means[larger, 3]
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
