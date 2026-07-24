from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.rgc_response import CellMetadata, RGCResponseSession, ResponseTargetKind


class SyntheticTeacherError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SyntheticTeacherResult:
    session: RGCResponseSession
    kernels: dict[str, np.ndarray]


def generate_teacher_responses(
    cone_sequences: np.ndarray,
    cone_positions_degs: np.ndarray,
    source_ids: tuple[str, ...],
    time_axis_seconds: np.ndarray,
    *,
    trials: int,
    seed: int,
    adaptive: bool,
) -> SyntheticTeacherResult:
    if cone_sequences.ndim != 3 or trials < 1:
        raise SyntheticTeacherError(
            "cone_sequences must be [stimulus,time,cone] and trials positive"
        )
    rng = np.random.default_rng(seed)
    stimulus_count, time_count, cone_count = cone_sequences.shape
    cell_count = min(4, cone_count)
    center_indices = np.linspace(0, cone_count - 1, cell_count, dtype=int)
    cells = CellMetadata(
        ids=tuple(f"synthetic-{index}" for index in range(cell_count)),
        type_ids=tuple(
            "midget" if index % 2 == 0 else "parasol"
            for index in range(cell_count)
        ),
        polarities=np.arange(cell_count, dtype=np.int64) % 2,
        positions_degs=cone_positions_degs[center_indices].astype(np.float32),
        eccentricities_deg=np.full(cell_count, 4.0, dtype=np.float32),
    )
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
    logits = _causal_logits(paired_cones, kernels)
    low_scale = np.ones(cell_count, dtype=np.float32)
    high_scale = (
        np.linspace(0.7, 1.3, cell_count, dtype=np.float32)
        if adaptive
        else low_scale.copy()
    )
    context_scale = np.stack(
        [low_scale if context == "low" else high_scale for context in contexts]
    )
    if adaptive:
        logits = logits * _adaptation_envelope(
            context_scale,
            time_count,
        )[:, :, None]
    probabilities = 1 / (1 + np.exp(-np.clip(logits - 2.0, -20.0, 20.0)))
    spikes = rng.binomial(
        1,
        probabilities[:, None, :, :],
        size=(len(contexts), trials, time_count, cell_count),
    ).astype(np.float32)
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
    )
    kernel_low = kernels.copy()
    kernel_high = kernels * high_scale[:, None, None]
    return SyntheticTeacherResult(
        session=session,
        kernels={
            "static_kernel": kernels,
            "context_kernel_low": kernel_low,
            "context_kernel_high": kernel_high,
        },
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


def _causal_logits(cones: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    stimulus_count, time_count, _ = cones.shape
    logits = np.zeros((stimulus_count, time_count, kernels.shape[0]), dtype=np.float32)
    centered = cones - cones.mean(axis=(0, 1), keepdims=True)
    centered /= cones.std(axis=(0, 1), keepdims=True) + 1e-6
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
    envelope = np.ones((scales.shape[0], time_count), dtype=np.float32)
    start = max(1, time_count - min(64, time_count // 2))
    for time in range(start, time_count):
        recovery = np.exp(-(time - start) / 30.0)
        envelope[:, time] = 1 + (scales.mean(axis=1) - 1) * recovery
    return envelope


__all__ = [
    "SyntheticTeacherError",
    "SyntheticTeacherResult",
    "generate_teacher_responses",
]
