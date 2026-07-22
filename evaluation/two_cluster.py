from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterResult:
    assignments: np.ndarray
    silhouette: float
    minimum_fraction: float
    means: np.ndarray
    separation: float


def cluster_two(features: np.ndarray, seed: int) -> ClusterResult:
    assignments = _kmeans_two(features, seed)
    fractions = np.bincount(assignments, minlength=2) / assignments.size
    if fractions.min() == 0.0:
        return ClusterResult(
            assignments,
            0.0,
            0.0,
            np.zeros((2, features.shape[1])),
            0.0,
        )
    means = np.stack(
        [features[assignments == cluster].mean(axis=0) for cluster in range(2)]
    )
    return ClusterResult(
        assignments=assignments,
        silhouette=_silhouette(features, assignments),
        minimum_fraction=float(fractions.min()),
        means=means,
        separation=float(np.linalg.norm(means[0] - means[1])),
    )


def _kmeans_two(features: np.ndarray, seed: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("k-means requires at least two eligible units")
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


__all__ = ["ClusterResult", "cluster_two"]
