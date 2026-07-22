from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import torch


HUMRET_SPLIT_HALF_SEED: Final = 7
HUMRET_SPLIT_HALF_ITERATIONS: Final = 1000
HUMRET_GRATING_SHAPE: Final = (6, 4)


@dataclass(frozen=True, slots=True)
class HumRetDataError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class FunctionalPopulationResponse:
    grating_f1_normalized: torch.Tensor
    chirp_modulation_normalized: torch.Tensor


@dataclass(frozen=True, slots=True)
class GratingPopulationDistances:
    mean_tuning_cosine_distance: float
    spatial_preference_total_variation: float
    temporal_preference_total_variation: float


@dataclass(frozen=True, slots=True)
class PopulationMetricResult:
    model_human_distance: float
    human_split_half_p95: float
    passed: bool


@dataclass(frozen=True, slots=True)
class HumRetFunctionalAgreement:
    grating_mean_tuning_cosine_distance: PopulationMetricResult
    grating_spatial_preference_total_variation: PopulationMetricResult
    grating_temporal_preference_total_variation: PopulationMetricResult
    chirp_mean_waveform_cosine_distance: PopulationMetricResult
    chirp_peak_frequency_total_variation: PopulationMetricResult
    external_functional_pass: bool
    bootstrap_seed: int
    bootstrap_iterations: int
    interpretation: Literal["functional_population_distribution_only"]


def parse_functional_population(
    grating_f1: np.ndarray | torch.Tensor,
    chirp_modulation: np.ndarray | torch.Tensor,
    label: str,
) -> FunctionalPopulationResponse:
    grating = _normalized_grating(grating_f1, f"{label}.grating_f1")
    chirp = torch.as_tensor(chirp_modulation, dtype=torch.float32).detach().clone()
    if chirp.ndim != 2 or chirp.shape[0] < 1 or chirp.shape[1] < 2:
        raise HumRetDataError(
            f"{label}.chirp_modulation must have shape [cell,frequency]"
        )
    if not torch.isfinite(chirp).all():
        raise HumRetDataError(f"{label}.chirp_modulation must be finite")
    scale = chirp.abs().amax(dim=1)
    if torch.any(scale <= 0):
        raise HumRetDataError(
            f"{label}.chirp_modulation contains a cell without modulation"
        )
    return FunctionalPopulationResponse(grating, chirp / scale[:, None])


def grating_population_distances(
    model_tuning: np.ndarray | torch.Tensor,
    human_tuning: np.ndarray | torch.Tensor,
) -> GratingPopulationDistances:
    model = _normalized_grating(model_tuning, "model_tuning")
    human = _normalized_grating(human_tuning, "human_tuning")
    distance, spatial, temporal = _grating_distances(model, human)
    return GratingPopulationDistances(distance, spatial, temporal)


def compare_functional_populations(
    model: FunctionalPopulationResponse,
    human: FunctionalPopulationResponse,
) -> HumRetFunctionalAgreement:
    human_cell_counts = (
        human.grating_f1_normalized.shape[0],
        human.chirp_modulation_normalized.shape[0],
    )
    if min(human_cell_counts) < 2:
        raise HumRetDataError(
            "human grating_f1 and chirp_modulation need at least two cells"
        )
    model_distances = _distance_vector(model, human)
    split_distances = np.empty((HUMRET_SPLIT_HALF_ITERATIONS, 5), dtype=np.float64)
    rng = np.random.default_rng(HUMRET_SPLIT_HALF_SEED)
    for iteration in range(HUMRET_SPLIT_HALF_ITERATIONS):
        grating_order = torch.from_numpy(
            rng.permutation(human.grating_f1_normalized.shape[0])
        )
        chirp_order = torch.from_numpy(
            rng.permutation(human.chirp_modulation_normalized.shape[0])
        )
        grating_midpoint = grating_order.numel() // 2
        chirp_midpoint = chirp_order.numel() // 2
        left = FunctionalPopulationResponse(
            human.grating_f1_normalized[grating_order[:grating_midpoint]],
            human.chirp_modulation_normalized[chirp_order[:chirp_midpoint]],
        )
        right = FunctionalPopulationResponse(
            human.grating_f1_normalized[grating_order[grating_midpoint:]],
            human.chirp_modulation_normalized[chirp_order[chirp_midpoint:]],
        )
        split_distances[iteration] = _distance_vector(left, right)
    metrics = tuple(
        PopulationMetricResult(
            model_human_distance=float(model_distances[index]),
            human_split_half_p95=float(np.percentile(split_distances[:, index], 95)),
            passed=bool(
                model_distances[index]
                <= np.percentile(split_distances[:, index], 95)
            ),
        )
        for index in range(5)
    )
    return HumRetFunctionalAgreement(
        grating_mean_tuning_cosine_distance=metrics[0],
        grating_spatial_preference_total_variation=metrics[1],
        grating_temporal_preference_total_variation=metrics[2],
        chirp_mean_waveform_cosine_distance=metrics[3],
        chirp_peak_frequency_total_variation=metrics[4],
        external_functional_pass=all(metric.passed for metric in metrics),
        bootstrap_seed=HUMRET_SPLIT_HALF_SEED,
        bootstrap_iterations=HUMRET_SPLIT_HALF_ITERATIONS,
        interpretation="functional_population_distribution_only",
    )


def _normalized_grating(
    tuning: np.ndarray | torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(tuning, dtype=torch.float32).detach().clone()
    if value.ndim != 3 or tuple(value.shape[1:]) != HUMRET_GRATING_SHAPE:
        raise HumRetDataError(f"{name} must have shape [cell,6,4]")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise HumRetDataError(f"{name} must be finite and non-negative")
    maximum = value.flatten(1).amax(dim=1)
    if torch.any(maximum <= 0):
        raise HumRetDataError(f"{name} contains a cell without grating response")
    return value / maximum[:, None, None]


def _distance_vector(
    left: FunctionalPopulationResponse,
    right: FunctionalPopulationResponse,
) -> np.ndarray:
    grating = _grating_distances(
        left.grating_f1_normalized,
        right.grating_f1_normalized,
    )
    chirp_cosine = _cosine_distance(
        left.chirp_modulation_normalized.mean(dim=0),
        right.chirp_modulation_normalized.mean(dim=0),
        "chirp_modulation",
    )
    bins = left.chirp_modulation_normalized.shape[1]
    left_peaks = _histogram(left.chirp_modulation_normalized.abs().argmax(dim=1), bins)
    right_peaks = _histogram(right.chirp_modulation_normalized.abs().argmax(dim=1), bins)
    return np.asarray(
        (*grating, chirp_cosine, _total_variation(left_peaks, right_peaks)),
        dtype=np.float64,
    )


def _grating_distances(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float, float]:
    cosine = _cosine_distance(left.mean(dim=0), right.mean(dim=0), "grating_f1")
    left_spatial, left_temporal = _grating_preferences(left)
    right_spatial, right_temporal = _grating_preferences(right)
    return (
        cosine,
        _total_variation(left_spatial, right_spatial),
        _total_variation(left_temporal, right_temporal),
    )


def _grating_preferences(tuning: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preferred = tuning.flatten(1).argmax(dim=1)
    temporal_bins = HUMRET_GRATING_SHAPE[1]
    return (
        _histogram(preferred // temporal_bins, HUMRET_GRATING_SHAPE[0]),
        _histogram(preferred % temporal_bins, temporal_bins),
    )


def _histogram(indices: torch.Tensor, bins: int) -> torch.Tensor:
    counts = torch.bincount(indices, minlength=bins).float()
    return counts / counts.sum()


def _cosine_distance(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    left_flat = left.flatten()
    right_flat = right.flatten()
    if left_flat.norm() <= 0 or right_flat.norm() <= 0:
        raise HumRetDataError(f"{name} population mean cannot be all zero")
    similarity = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=0)
    return max(0.0, 1.0 - float(similarity.item()))


def _total_variation(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((0.5 * torch.abs(left - right).sum()).item())
