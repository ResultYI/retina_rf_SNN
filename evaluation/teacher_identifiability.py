from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.point_process_teacher import (
    _SPIKE_HISTORY_DECAY,
    _SPIKE_HISTORY_LOGIT_GAIN,
    _causal_logits,
)
from data.rgc_response import load_rgc_response
from data.synthetic_teacher import (
    load_teacher_input_normalization,
    load_teacher_rf_metadata,
)


class TeacherIdentifiabilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TeacherTargets:
    expected_probabilities: np.ndarray
    conditional_probabilities: np.ndarray
    teacher_signed_gains: np.ndarray
    spike_counts: np.ndarray
    source_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    type_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CellIdentifiability:
    cell_id: str
    type_id: str
    teacher_signed_gain: float
    expected_probability_gain: float
    psth_gain: float
    psth_gain_ci: tuple[float, float]
    expected_direction_matches: bool
    psth_direction_matches: bool
    psth_ci_supports_direction: bool
    changed_spike_bin_fraction: float


@dataclass(frozen=True, slots=True)
class EmpiricalIdentifiabilityAudit:
    source_pair_count: int
    trial_count: int
    probe_steps: int
    expected_direction_count: int
    psth_direction_count: int
    psth_ci_supported_count: int
    changed_spike_bin_fraction: float
    cells: tuple[CellIdentifiability, ...]


def reconstruct_teacher_targets(path: str | Path) -> TeacherTargets:
    session = load_rgc_response(path)
    normalization = load_teacher_input_normalization(
        path,
        session.cone_response.shape[2],
    )
    metadata = load_teacher_rf_metadata(path)
    if normalization is None or metadata is None:
        raise TeacherIdentifiabilityError("Synthetic teacher metadata is required")
    envelope = metadata.context_gain_envelope
    if envelope is None or envelope.shape != (
        session.cone_response.shape[0],
        session.cone_response.shape[1],
        session.spike_counts.shape[3],
    ):
        raise TeacherIdentifiabilityError("Teacher context envelope shape is invalid")
    logits = _causal_logits(
        session.cone_response,
        metadata.static_kernel,
        normalization,
    ) * envelope
    base_logits = logits - np.float32(2.0)
    expected = _sigmoid(base_logits)
    conditional = _conditional_probabilities(base_logits, session.spike_counts)
    low_norm = np.linalg.norm(metadata.context_kernel_low.reshape(len(session.cells.ids), -1), axis=1)
    high_norm = np.linalg.norm(metadata.context_kernel_high.reshape(len(session.cells.ids), -1), axis=1)
    signed_gains = np.log((high_norm + 1e-8) / (low_norm + 1e-8))
    return TeacherTargets(
        expected.astype(np.float32),
        conditional,
        signed_gains.astype(np.float32),
        session.spike_counts,
        session.source_ids,
        session.context_ids,
        session.cells.ids,
        session.cells.type_ids,
    )


def audit_empirical_identifiability(
    path: str | Path,
    *,
    probe_steps: int,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> EmpiricalIdentifiabilityAudit:
    targets = reconstruct_teacher_targets(path)
    pairs = _context_pairs(targets.source_ids, targets.context_ids)
    if not pairs or not 1 <= probe_steps <= targets.spike_counts.shape[2]:
        raise TeacherIdentifiabilityError("Matched pairs and a valid probe are required")
    expected_deltas = []
    psth_deltas = []
    changed = []
    trial_deltas = []
    for low, high in pairs:
        expected_deltas.append(
            targets.expected_probabilities[high, -probe_steps:].mean(axis=0)
            - targets.expected_probabilities[low, -probe_steps:].mean(axis=0)
        )
        psth_deltas.append(
            targets.spike_counts[high, :, -probe_steps:].mean(axis=(0, 1))
            - targets.spike_counts[low, :, -probe_steps:].mean(axis=(0, 1))
        )
        difference = (
            targets.spike_counts[high, :, -probe_steps:]
            != targets.spike_counts[low, :, -probe_steps:]
        )
        changed.append(difference.mean(axis=(0, 1)))
        trial_deltas.append(
            (
                targets.spike_counts[high, :, -probe_steps:]
                - targets.spike_counts[low, :, -probe_steps:]
            ).mean(axis=1)
        )
    expected_gain = np.stack(expected_deltas).mean(axis=0)
    psth_gain = np.stack(psth_deltas).mean(axis=0)
    changed_fraction = np.stack(changed).mean(axis=0)
    replicate_deltas = np.concatenate(trial_deltas, axis=0)
    cis = _bootstrap_cis(replicate_deltas, bootstrap_iterations, seed)
    cells = tuple(
        _cell_result(
            targets,
            index,
            expected_gain,
            psth_gain,
            cis,
            changed_fraction,
        )
        for index in range(len(targets.cell_ids))
    )
    return EmpiricalIdentifiabilityAudit(
        source_pair_count=len(pairs),
        trial_count=targets.spike_counts.shape[1],
        probe_steps=probe_steps,
        expected_direction_count=sum(cell.expected_direction_matches for cell in cells),
        psth_direction_count=sum(cell.psth_direction_matches for cell in cells),
        psth_ci_supported_count=sum(cell.psth_ci_supports_direction for cell in cells),
        changed_spike_bin_fraction=float(changed_fraction.mean()),
        cells=cells,
    )


def _conditional_probabilities(
    base_logits: np.ndarray,
    spikes: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros_like(spikes, dtype=np.float32)
    history = np.zeros(spikes.shape[:2] + (spikes.shape[3],), dtype=np.float32)
    for time in range(spikes.shape[2]):
        probabilities[:, :, time] = _sigmoid(
            base_logits[:, None, time] + _SPIKE_HISTORY_LOGIT_GAIN * history
        )
        history = _SPIKE_HISTORY_DECAY * history + spikes[:, :, time]
    return probabilities


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -20.0, 20.0)))


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


def _bootstrap_cis(
    replicate_deltas: np.ndarray,
    iterations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        replicate_deltas.shape[0],
        size=(max(1, iterations), replicate_deltas.shape[0]),
    )
    means = replicate_deltas[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975), axis=0).T


def _cell_result(
    targets: TeacherTargets,
    index: int,
    expected_gain: np.ndarray,
    psth_gain: np.ndarray,
    cis: np.ndarray,
    changed_fraction: np.ndarray,
) -> CellIdentifiability:
    direction = np.sign(targets.teacher_signed_gains[index])
    low, high = float(cis[index, 0]), float(cis[index, 1])
    return CellIdentifiability(
        targets.cell_ids[index],
        targets.type_ids[index],
        float(targets.teacher_signed_gains[index]),
        float(expected_gain[index]),
        float(psth_gain[index]),
        (low, high),
        bool(np.sign(expected_gain[index]) == direction),
        bool(np.sign(psth_gain[index]) == direction),
        bool(low > 0 if direction > 0 else high < 0),
        float(changed_fraction[index]),
    )


__all__ = [
    "CellIdentifiability",
    "EmpiricalIdentifiabilityAudit",
    "TeacherIdentifiabilityError",
    "TeacherTargets",
    "audit_empirical_identifiability",
    "reconstruct_teacher_targets",
]
