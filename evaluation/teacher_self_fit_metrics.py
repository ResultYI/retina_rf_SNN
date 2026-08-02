from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GroupEffectRecovery:
    effect: str
    estimate_mean: float
    teacher_value: float
    error: float
    bootstrap_interval: tuple[float, float]
    recovery_power: float


@dataclass(frozen=True, slots=True)
class TeacherSelfFitPoint:
    trial_count: int
    fit_trial_count: int
    heldout_trial_count: int
    optimizer_evaluations: int
    direction_recovery_rate: float
    per_cell_direction_recovery_rate: tuple[float, ...]
    context_gain_confidence_intervals: tuple[tuple[float, float], ...]
    context_gain_ci_supported_count: int
    per_cell_kernel_error: tuple[float, ...]
    per_cell_kernel_correlation: tuple[float, ...]
    group_effect_recovery: tuple[GroupEffectRecovery, ...]
    teacher_kernel_correlation_mean: float
    per_cell_context_kernel_correlation: tuple[float, ...]
    heldout_nll_mean: float
    teacher_oracle_nll_mean: float
    excess_nll_mean: float
    history_gain_mean: tuple[float, ...]
    history_gain_mae: float


@dataclass(frozen=True, slots=True)
class TeacherSelfFitSummaryInput:
    trial_count: int
    fit_trial_count: int
    optimizer_evaluations: int
    recovered_low_kernel: np.ndarray
    recovered_high_kernel: np.ndarray
    recovered_context_kernel: np.ndarray
    teacher_low_kernel: np.ndarray
    teacher_high_kernel: np.ndarray
    heldout_nll: np.ndarray
    teacher_oracle_nll: np.ndarray
    history_gain: np.ndarray
    true_history_gain: float
    type_ids: tuple[str, ...]
    polarities: np.ndarray


def summarize_teacher_self_fit(
    values: TeacherSelfFitSummaryInput,
) -> TeacherSelfFitPoint:
    gains = signed_gains(values.recovered_low_kernel, values.recovered_high_kernel)
    target = signed_gains(values.teacher_low_kernel, values.teacher_high_kernel)
    directions = np.sign(gains) == np.sign(target)[None]
    intervals = np.quantile(gains, (0.025, 0.975), axis=0).T
    ci_supported = np.where(
        target > 0,
        intervals[:, 0] > 0,
        intervals[:, 1] < 0,
    )
    context_correlations = _kernel_cosine(
        values.recovered_context_kernel,
        values.teacher_high_kernel - values.teacher_low_kernel,
    )
    low_correlation = _kernel_cosine(
        values.recovered_low_kernel,
        values.teacher_low_kernel,
    )
    high_correlation = _kernel_cosine(
        values.recovered_high_kernel,
        values.teacher_high_kernel,
    )
    kernel_correlation = (low_correlation + high_correlation) / 2
    kernel_error = _kernel_error(
        values.recovered_low_kernel,
        values.recovered_high_kernel,
        values.teacher_low_kernel,
        values.teacher_high_kernel,
    )
    return TeacherSelfFitPoint(
        values.trial_count,
        values.fit_trial_count,
        values.trial_count - values.fit_trial_count,
        values.optimizer_evaluations,
        float(directions.all(axis=1).mean()),
        tuple(float(value) for value in directions.mean(axis=0)),
        tuple((float(low), float(high)) for low, high in intervals),
        int(ci_supported.sum()),
        tuple(float(value) for value in kernel_error.mean(axis=0)),
        tuple(float(value) for value in kernel_correlation.mean(axis=0)),
        _group_effect_recovery(gains, target, values.type_ids, values.polarities),
        float(kernel_correlation.mean()),
        tuple(float(value) for value in context_correlations.mean(axis=0)),
        float(values.heldout_nll.mean()),
        float(values.teacher_oracle_nll.mean()),
        float((values.heldout_nll - values.teacher_oracle_nll).mean()),
        tuple(float(value) for value in values.history_gain.mean(axis=0)),
        float(np.abs(values.history_gain - values.true_history_gain).mean()),
    )


def signed_gains(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low_norm = np.linalg.norm(low.reshape(*low.shape[:-2], -1), axis=-1)
    high_norm = np.linalg.norm(high.reshape(*high.shape[:-2], -1), axis=-1)
    return np.log((high_norm + 1e-8) / (low_norm + 1e-8))


def _kernel_error(
    recovered_low: np.ndarray,
    recovered_high: np.ndarray,
    teacher_low: np.ndarray,
    teacher_high: np.ndarray,
) -> np.ndarray:
    low_delta = (recovered_low - teacher_low[None]).reshape(
        *recovered_low.shape[:-2],
        -1,
    )
    high_delta = (recovered_high - teacher_high[None]).reshape(
        *recovered_high.shape[:-2],
        -1,
    )
    low_error = (low_delta**2).mean(axis=-1)
    high_error = (high_delta**2).mean(axis=-1)
    return (low_error + high_error) / 2


def _kernel_cosine(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted_flat = predicted.reshape(*predicted.shape[:-2], -1)
    target_flat = target.reshape(*target.shape[:-2], -1)
    numerator = (predicted_flat * target_flat).sum(axis=-1)
    denominator = np.linalg.norm(predicted_flat, axis=-1) * np.linalg.norm(
        target_flat,
        axis=-1,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _group_effect_recovery(
    gains: np.ndarray,
    target: np.ndarray,
    type_ids: tuple[str, ...],
    polarities: np.ndarray,
) -> tuple[GroupEffectRecovery, ...]:
    estimates = _factorial_effects(gains, type_ids, polarities)
    teacher = _factorial_effects(target[None], type_ids, polarities)
    if estimates.shape[1] != 3 or teacher.shape[1] != 3:
        return ()
    intervals = np.quantile(estimates, (0.025, 0.975), axis=0).T
    names = ("type", "polarity", "interaction")
    return tuple(
        GroupEffectRecovery(
            names[index],
            float(estimates[:, index].mean()),
            float(teacher[0, index]),
            float(abs(estimates[:, index].mean() - teacher[0, index])),
            (float(intervals[index, 0]), float(intervals[index, 1])),
            _effect_power(estimates[:, index], float(teacher[0, index])),
        )
        for index in range(3)
    )


def _factorial_effects(
    gains: np.ndarray,
    type_ids: tuple[str, ...],
    polarities: np.ndarray,
) -> np.ndarray:
    labels = tuple(sorted(set(type_ids)))
    polarity_values = tuple(sorted(int(value) for value in set(polarities.tolist())))
    if len(labels) != 2 or polarity_values != (0, 1):
        return np.zeros((gains.shape[0], 0), dtype=np.float64)
    means = []
    for type_id in labels:
        for polarity in polarity_values:
            mask = np.asarray(
                [
                    cell_type == type_id and int(cell_polarity) == polarity
                    for cell_type, cell_polarity in zip(
                        type_ids,
                        polarities,
                        strict=True,
                    )
                ],
                dtype=bool,
            )
            if not mask.any():
                return np.zeros((gains.shape[0], 0), dtype=np.float64)
            means.append(gains[:, mask].mean(axis=1))
    on_first, off_first, on_second, off_second = means
    return np.stack(
        (
            (-on_first - off_first + on_second + off_second) / 4,
            (-on_first + off_first - on_second + off_second) / 4,
            (on_first - off_first - on_second + off_second) / 4,
        ),
        axis=1,
    )


def _effect_power(values: np.ndarray, target: float) -> float:
    if abs(target) <= 1e-8:
        return float((np.abs(values) <= 1e-8).mean())
    return float((np.sign(values) == np.sign(target)).mean())


__all__ = [
    "GroupEffectRecovery",
    "TeacherSelfFitPoint",
    "TeacherSelfFitSummaryInput",
    "signed_gains",
    "summarize_teacher_self_fit",
]
