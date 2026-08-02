from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.point_process_teacher import _sample_history_conditioned_spikes
from evaluation.teacher_identifiability import reconstruct_teacher_targets


class TrialPowerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrialPowerPoint:
    trial_count: int
    direction_recovery_rate: float
    ci_support_rate: float
    per_cell_conditional_gain: tuple[float, ...]
    per_cell_direction_recovery_rate: tuple[float, ...]
    per_cell_ci_support_rate: tuple[float, ...]
    per_cell_gain_bias: tuple[float, ...]
    per_cell_gain_variance: tuple[float, ...]
    direction_gate_passed: bool
    ci_gate_passed: bool


@dataclass(frozen=True, slots=True)
class TrialPowerCurve:
    monte_carlo_seeds: int
    bootstrap_iterations: int
    probe_steps: int
    direction_gate: float
    ci_gate: float
    minimum_passing_trial_count: int | None
    points: tuple[TrialPowerPoint, ...]


@dataclass(frozen=True, slots=True)
class TrialPowerRequest:
    path: str | Path
    trial_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    monte_carlo_seeds: int = 100
    bootstrap_iterations: int = 500
    probe_steps: int = 64
    seed: int = 0
    direction_gate: float = 0.90
    ci_gate: float = 0.80


@dataclass(frozen=True, slots=True)
class _PowerPointInput:
    base_logits: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    directions: np.ndarray
    trial_count: int
    settings: TrialPowerRequest
    seed: int


@dataclass(frozen=True, slots=True)
class _GainEstimateInput:
    spikes: np.ndarray
    probabilities: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    probe_steps: int
    bootstrap_iterations: int
    seed: int


@dataclass(frozen=True, slots=True)
class _GainEstimate:
    empirical: np.ndarray
    conditional: np.ndarray
    confidence_intervals: np.ndarray


def audit_trial_power_curve(request: TrialPowerRequest) -> TrialPowerCurve:
    targets = reconstruct_teacher_targets(request.path)
    if (
        not request.trial_counts
        or any(count < 1 for count in request.trial_counts)
        or request.monte_carlo_seeds < 1
        or request.bootstrap_iterations < 1
        or not 1 <= request.probe_steps <= targets.spike_counts.shape[2]
        or not 0 <= request.direction_gate <= 1
        or not 0 <= request.ci_gate <= 1
    ):
        raise TrialPowerError("Trial power settings are invalid")
    probabilities = np.clip(targets.expected_probabilities, 1e-7, 1 - 1e-7)
    base_logits = np.log(probabilities / (1 - probabilities))
    pairs = _context_pairs(targets.source_ids, targets.context_ids)
    directions = np.sign(targets.teacher_signed_gains)
    points = tuple(
        _power_point(
            _PowerPointInput(
                base_logits,
                pairs,
                directions,
                trial_count,
                request,
                request.seed + index * 100_000,
            )
        )
        for index, trial_count in enumerate(request.trial_counts)
    )
    passing = tuple(
        point.trial_count
        for point in points
        if point.direction_gate_passed and point.ci_gate_passed
    )
    return TrialPowerCurve(
        request.monte_carlo_seeds,
        request.bootstrap_iterations,
        request.probe_steps,
        request.direction_gate,
        request.ci_gate,
        min(passing) if passing else None,
        points,
    )


def _power_point(values: _PowerPointInput) -> TrialPowerPoint:
    estimates = []
    conditional = []
    recovered = []
    supported = []
    recovered_by_cell = []
    supported_by_cell = []
    for offset in range(values.settings.monte_carlo_seeds):
        spikes, probabilities = _sample_history_conditioned_spikes(
            np.random.default_rng(values.seed + offset),
            values.base_logits,
            values.trial_count,
        )
        estimate = _estimate_gain(
            _GainEstimateInput(
                spikes,
                probabilities,
                values.pairs,
                values.settings.probe_steps,
                values.settings.bootstrap_iterations,
                values.seed + offset,
            )
        )
        estimates.append(estimate.empirical)
        conditional.append(estimate.conditional)
        direction_matches = np.sign(estimate.empirical) == values.directions
        ci_matches = _ci_support_mask(
            estimate.confidence_intervals,
            values.directions,
        )
        recovered.append(bool(np.all(direction_matches)))
        supported.append(bool(np.all(ci_matches)))
        recovered_by_cell.append(direction_matches)
        supported_by_cell.append(ci_matches)
    empirical_values = np.stack(estimates)
    conditional_values = np.stack(conditional)
    direction_rate = float(np.mean(recovered))
    ci_rate = float(np.mean(supported))
    return TrialPowerPoint(
        values.trial_count,
        direction_rate,
        ci_rate,
        tuple(float(value) for value in conditional_values.mean(axis=0)),
        tuple(float(value) for value in np.stack(recovered_by_cell).mean(axis=0)),
        tuple(float(value) for value in np.stack(supported_by_cell).mean(axis=0)),
        tuple(float(value) for value in (empirical_values - conditional_values).mean(axis=0)),
        tuple(float(value) for value in empirical_values.var(axis=0)),
        direction_rate >= values.settings.direction_gate,
        ci_rate >= values.settings.ci_gate,
    )


def _estimate_gain(values: _GainEstimateInput) -> _GainEstimate:
    empirical = []
    conditional = []
    replicates = []
    for low, high in values.pairs:
        empirical.append(
            values.spikes[high, :, -values.probe_steps:].mean(axis=(0, 1))
            - values.spikes[low, :, -values.probe_steps:].mean(axis=(0, 1))
        )
        conditional.append(
            values.probabilities[high, :, -values.probe_steps:].mean(axis=(0, 1))
            - values.probabilities[low, :, -values.probe_steps:].mean(axis=(0, 1))
        )
        replicates.append(
            (
                values.spikes[high, :, -values.probe_steps:]
                - values.spikes[low, :, -values.probe_steps:]
            ).mean(axis=1)
        )
    replicate_values = np.concatenate(replicates, axis=0)
    rng = np.random.default_rng(values.seed)
    indices = rng.integers(
        0,
        replicate_values.shape[0],
        size=(values.bootstrap_iterations, replicate_values.shape[0]),
    )
    bootstrap_means = replicate_values[indices].mean(axis=1)
    confidence_intervals = np.quantile(bootstrap_means, (0.025, 0.975), axis=0).T
    return _GainEstimate(
        np.stack(empirical).mean(axis=0),
        np.stack(conditional).mean(axis=0),
        confidence_intervals,
    )


def _context_pairs(
    source_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    grouped: dict[str, dict[str, int]] = {}
    for index, (source, context) in enumerate(zip(source_ids, context_ids, strict=True)):
        grouped.setdefault(source, {})[context] = index
    return tuple(
        (contexts["low"], contexts["high"])
        for contexts in grouped.values()
        if "low" in contexts and "high" in contexts
    )


def _ci_support_mask(
    confidence_intervals: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    return np.where(
        directions > 0,
        confidence_intervals[:, 0] > 0,
        confidence_intervals[:, 1] < 0,
    )


__all__ = [
    "TrialPowerCurve",
    "TrialPowerError",
    "TrialPowerPoint",
    "TrialPowerRequest",
    "audit_trial_power_curve",
]
