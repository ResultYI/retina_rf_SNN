from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


class Phase1ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetCondition:
    stimulus_count: int
    trials_per_stimulus: int

    def __post_init__(self) -> None:
        if self.stimulus_count < 2 or self.stimulus_count % 2:
            raise Phase1ConfigurationError(
                "stimulus_count must be a positive matched-context pair count"
            )
        if self.trials_per_stimulus < 1:
            raise Phase1ConfigurationError("trials_per_stimulus must be positive")
        if self.sequence_budget != 896:
            raise Phase1ConfigurationError(
                "every phase-one condition must contain exactly 896 sequences"
            )

    @property
    def label(self) -> str:
        return f"{self.stimulus_count}x{self.trials_per_stimulus}"

    @property
    def source_count(self) -> int:
        return self.stimulus_count // 2

    @property
    def sequence_budget(self) -> int:
        return self.stimulus_count * self.trials_per_stimulus


@dataclass(frozen=True, slots=True)
class NestedStimulusBank:
    cone_sequences: np.ndarray
    source_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    original_source_count: int
    generator: str = "empirical_log_cone_var1_v1"

    def __post_init__(self) -> None:
        if self.cone_sequences.ndim != 3:
            raise Phase1ConfigurationError(
                "cone_sequences must have shape [source,time,cone]"
            )
        if self.cone_sequences.shape[0] != len(self.source_ids):
            raise Phase1ConfigurationError("source ids do not match cone sequences")
        if len(self.source_ids) != len(self.source_sha256):
            raise Phase1ConfigurationError("source hashes do not match cone sequences")


def generate_nested_stimulus_bank(
    reference_cones: np.ndarray,
    source_ids: tuple[str, ...],
    source_count: int,
    *,
    seed: int,
) -> NestedStimulusBank:
    cones = np.asarray(reference_cones, dtype=np.float32)
    _validate_reference_cones(cones, source_ids, source_count)
    if source_count == cones.shape[0]:
        return _stimulus_bank(cones.copy(), source_ids, cones.shape[0])
    logarithmic = np.log1p(cones.astype(np.float64))
    flattened = logarithmic.reshape(-1, logarithmic.shape[-1])
    mean = flattened.mean(axis=0)
    centered_previous = (logarithmic[:, :-1] - mean).reshape(
        -1, logarithmic.shape[-1]
    )
    centered_next = (logarithmic[:, 1:] - mean).reshape(
        -1, logarithmic.shape[-1]
    )
    ridge = max(1e-8, float(np.trace(centered_previous.T @ centered_previous)) * 1e-8)
    transition = np.linalg.solve(
        centered_previous.T @ centered_previous
        + ridge * np.eye(centered_previous.shape[1]),
        centered_previous.T @ centered_next,
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if spectral_radius > 0.98:
        transition *= 0.98 / spectral_radius
    residual = centered_next - centered_previous @ transition
    covariance = np.cov(residual, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * max(1e-10, float(np.trace(covariance)) * 1e-8)
    cholesky = np.linalg.cholesky(covariance)
    lower = np.quantile(logarithmic, 0.001, axis=(0, 1))
    upper = np.quantile(logarithmic, 0.999, axis=(0, 1))
    rng = np.random.default_rng(seed)
    generated = [sequence.copy() for sequence in cones]
    generated_ids = list(source_ids)
    while len(generated) < source_count:
        latent = _sample_var1_sequence(
            logarithmic,
            mean,
            transition,
            cholesky,
            lower,
            upper,
            rng,
        )
        generated.append(np.expm1(latent).clip(min=0).astype(np.float32))
        generated_ids.append(f"var1-surrogate-{len(generated) - cones.shape[0]:03d}")
    return _stimulus_bank(
        np.stack(generated),
        tuple(generated_ids),
        cones.shape[0],
    )


def _sample_var1_sequence(
    logarithmic: np.ndarray,
    mean: np.ndarray,
    transition: np.ndarray,
    cholesky: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    time_count = logarithmic.shape[1]
    source_index = int(rng.integers(logarithmic.shape[0]))
    time_index = int(rng.integers(time_count))
    state = logarithmic[source_index, time_index].copy()
    values = np.empty((time_count, logarithmic.shape[2]), dtype=np.float64)
    for time in range(time_count):
        innovation = rng.standard_normal(cholesky.shape[0]) @ cholesky.T
        state = mean + (state - mean) @ transition + innovation
        state = np.clip(state, lower, upper)
        values[time] = state
    return values


def _validate_reference_cones(
    cones: np.ndarray,
    source_ids: tuple[str, ...],
    source_count: int,
) -> None:
    if cones.ndim != 3 or cones.shape[0] < 2:
        raise Phase1ConfigurationError(
            "reference cones must have shape [source,time,cone]"
        )
    if cones.shape[0] != len(source_ids):
        raise Phase1ConfigurationError("reference source ids do not match cones")
    if source_count < cones.shape[0]:
        raise Phase1ConfigurationError("source_count cannot drop canonical sources")
    if not np.isfinite(cones).all() or np.any(cones < 0):
        raise Phase1ConfigurationError("reference cones must be finite and non-negative")
    if not math.isfinite(float(cones.mean())):
        raise Phase1ConfigurationError("reference cone mean must be finite")


def _stimulus_bank(
    cones: np.ndarray,
    source_ids: tuple[str, ...],
    original_source_count: int,
) -> NestedStimulusBank:
    hashes = tuple(
        hashlib.sha256(np.ascontiguousarray(sequence).view(np.uint8)).hexdigest()
        for sequence in cones
    )
    return NestedStimulusBank(cones, source_ids, hashes, original_source_count)


__all__ = [
    "BudgetCondition",
    "NestedStimulusBank",
    "Phase1ConfigurationError",
    "generate_nested_stimulus_bank",
]
