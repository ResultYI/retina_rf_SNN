from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from data.input_identity import InputIdentity, synthetic_input_identity
from data.rgc_response import CellMetadata, RGCResponseSession, ResponseTargetKind
from data.synthetic_teacher import TeacherInputNormalization
from benchmarks.teacher_population import (
    SyntheticTeacherError,
    TeacherPopulationConfig,
    build_teacher_population,
    teacher_population_metadata,
)


_SPIKE_HISTORY_DECAY: Final = np.float32(np.exp(-0.1))
_SPIKE_HISTORY_LOGIT_GAIN: Final = np.float32(-1.5)


@dataclass(frozen=True, slots=True)
class SyntheticTeacherResult:
    session: RGCResponseSession
    kernels: dict[str, np.ndarray]
    expected_probabilities: np.ndarray
    conditional_probabilities: np.ndarray
    teacher_normalization: TeacherInputNormalization


def generate_teacher_responses(
    cone_sequences: np.ndarray,
    cone_positions_degs: np.ndarray,
    source_ids: tuple[str, ...],
    time_axis_seconds: np.ndarray,
    *,
    trials: int,
    seed: int,
    adaptive: bool,
    teacher_normalization: TeacherInputNormalization,
    input_identity: InputIdentity | None = None,
    population_config: TeacherPopulationConfig = TeacherPopulationConfig(),
    teacher_seed: int | None = None,
) -> SyntheticTeacherResult:
    if cone_sequences.ndim != 3 or trials < 1:
        raise SyntheticTeacherError(
            "cone_sequences must be [stimulus,time,cone] and trials positive"
        )
    rng = np.random.default_rng(seed)
    _, time_count, cone_count = cone_sequences.shape
    generation_seed = seed if teacher_seed is None else teacher_seed
    population = build_teacher_population(
        cone_positions_degs,
        population_config,
        generation_seed=generation_seed,
    )
    cells = population.cells
    cell_count = len(cells.ids)
    lag_count = min(16, time_count)
    kernels = _teacher_kernels(
        cone_positions_degs,
        cells,
        lag_count,
    )
    paired_cones, paired_sources, contexts = _matched_context_pairs(
        cone_sequences,
        source_ids,
    )
    identity = input_identity or synthetic_input_identity(cone_count, source_ids)
    if len(identity.stimulus_source_fingerprints) != len(source_ids):
        raise SyntheticTeacherError(
            "Input identity fingerprints must match synthetic source ids"
        )
    fingerprints = dict(
        zip(source_ids, identity.stimulus_source_fingerprints, strict=True)
    )
    paired_identity = identity.with_sources(
        tuple(fingerprints[source_id] for source_id in paired_sources),
        generator_name=f"{identity.generator_name}+point_process_teacher",
        generator_revision=f"{identity.generator_revision}+2",
    )
    logits = _causal_logits(paired_cones, kernels, teacher_normalization)
    low_scale = np.ones(cell_count, dtype=np.float32)
    high_scale = population.high_scales if adaptive else low_scale.copy()
    context_scale = np.stack(
        [low_scale if context == "low" else high_scale for context in contexts]
    )
    envelope = _adaptation_envelope(context_scale, time_count)
    if adaptive:
        logits = logits * envelope
    expected_probabilities = 1 / (
        1 + np.exp(-np.clip(logits - 2.0, -20.0, 20.0))
    )
    spikes, conditional_probabilities = _sample_history_conditioned_spikes(
        rng,
        logits - 2.0,
        trials,
    )
    session = RGCResponseSession(
        cone_response=paired_cones.astype(np.float32),
        spike_counts=spikes,
        valid_mask=np.ones_like(spikes, dtype=bool),
        time_axis_seconds=time_axis_seconds.astype(np.float64),
        cone_positions_degs=cone_positions_degs.astype(np.float32),
        cells=cells,
        source_ids=paired_sources,
        context_ids=contexts,
        target_kind=ResponseTargetKind.BERNOULLI,
        path=Path("<synthetic-memory>"),
        input_identity=paired_identity,
    )
    low_index = contexts.index("low")
    high_index = contexts.index("high")
    kernel_low = kernels * envelope[low_index, -1, :, None, None]
    kernel_high = kernels * envelope[high_index, -1, :, None, None]
    return SyntheticTeacherResult(
        session=session,
        kernels={
            "static_kernel": kernels,
            "context_kernel_low": kernel_low,
            "context_kernel_high": kernel_high,
            "context_gain_envelope": envelope,
            **teacher_population_metadata(population),
        },
        expected_probabilities=expected_probabilities,
        conditional_probabilities=conditional_probabilities,
        teacher_normalization=teacher_normalization,
    )


def _teacher_kernels(
    cone_positions: np.ndarray,
    cells: CellMetadata,
    lag_count: int,
) -> np.ndarray:
    distances = np.linalg.norm(
        cells.positions_degs[:, None, :] - cone_positions[None, :, :],
        axis=-1,
    )
    spatial = np.exp(-0.5 * np.square(distances / 0.08))
    spatial /= spatial.sum(axis=1, keepdims=True)
    polarity = np.where(cells.polarities == 0, 1.0, -1.0)[:, None]
    temporal = np.exp(-np.arange(lag_count, dtype=np.float32) / 4.0)
    temporal -= 0.25 * np.exp(-np.arange(lag_count, dtype=np.float32) / 10.0)
    return polarity[:, None, :] * temporal[None, :, None] * spatial[:, None, :]


def _causal_logits(
    cones: np.ndarray,
    kernels: np.ndarray,
    normalization: TeacherInputNormalization,
) -> np.ndarray:
    stimulus_count, time_count, _ = cones.shape
    logits = np.zeros((stimulus_count, time_count, kernels.shape[0]), dtype=np.float32)
    centered = normalization.normalize(cones)
    for lag in range(kernels.shape[1]):
        if lag >= time_count:
            break
        logits[:, lag:] += np.einsum(
            "stc,rc->str",
            centered[:, : time_count - lag],
            kernels[:, lag],
        )
    return logits


def _matched_context_pairs(
    cones: np.ndarray,
    source_ids: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    context_steps = max(1, cones.shape[1] - min(64, cones.shape[1] // 2))
    paired: list[np.ndarray] = []
    paired_sources: list[str] = []
    contexts: list[str] = []
    for sequence, source_id in zip(cones, source_ids, strict=True):
        for label, gain in (("low", 0.5), ("high", 1.5)):
            modified = sequence.copy()
            modified[:context_steps] *= gain
            paired.append(modified)
            paired_sources.append(source_id)
            contexts.append(label)
    return np.stack(paired), tuple(paired_sources), tuple(contexts)


def _adaptation_envelope(scales: np.ndarray, time_count: int) -> np.ndarray:
    envelope = np.ones(
        (scales.shape[0], time_count, scales.shape[1]),
        dtype=np.float32,
    )
    start = max(1, time_count - min(64, time_count // 2))
    for time in range(start, time_count):
        recovery = np.exp(-(time - start) / 30.0)
        envelope[:, time] = 1 + (scales - 1) * recovery
    return envelope


def _sample_history_conditioned_spikes(
    rng: np.random.Generator,
    base_logits: np.ndarray,
    trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    stimulus_count, time_count, cell_count = base_logits.shape
    shape = (stimulus_count, trials, time_count, cell_count)
    spikes = np.zeros(shape, dtype=np.float32)
    probabilities = np.zeros(shape, dtype=np.float32)
    history = np.zeros((stimulus_count, trials, cell_count), dtype=np.float32)
    for time in range(time_count):
        logits = (
            base_logits[:, None, time]
            + _SPIKE_HISTORY_LOGIT_GAIN * history
        )
        probability = 1 / (1 + np.exp(-np.clip(logits, -20.0, 20.0)))
        event = rng.binomial(1, probability).astype(np.float32)
        probabilities[:, :, time] = probability
        spikes[:, :, time] = event
        history = _SPIKE_HISTORY_DECAY * history + event
    return spikes, probabilities


__all__ = [
    "SyntheticTeacherError",
    "SyntheticTeacherResult",
    "TeacherPopulationConfig",
    "generate_teacher_responses",
]
