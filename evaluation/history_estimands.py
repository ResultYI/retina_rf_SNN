from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmarks.point_process_teacher import (
    _SPIKE_HISTORY_DECAY,
    _SPIKE_HISTORY_LOGIT_GAIN,
    _sample_history_conditioned_spikes,
)
from data.synthetic_teacher import load_teacher_rf_metadata
from evaluation.rf_history_contracts import (
    RFHistoryContract,
    RF_HISTORY_CONTRACTS,
    standard_train_rate_history_counts,
)
from evaluation.teacher_identifiability import reconstruct_teacher_targets
from evaluation.teacher_self_fit_model import history_trace
from training.response_data import ResponseSplit


class HistoryEstimandError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HistoryEstimandRequest:
    path: str | Path
    probe_steps: int = 64
    endogenous_trials: int = 2048
    seed: int = 0


@dataclass(frozen=True, slots=True)
class HistoryModeResult:
    history_contract: str
    history_source: str
    conditional_logit_rf_gain: tuple[float, ...]
    probability_rf_gain: tuple[float, ...]
    response_gain: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class HistoryEstimandAudit:
    probe_steps: int
    endogenous_trials: int
    cell_ids: tuple[str, ...]
    controlled_by_history: dict[RFHistoryContract, HistoryModeResult]
    modes: tuple[HistoryModeResult, ...]
    direct_response_gain: tuple[float, ...]
    history_mediated_gain: tuple[float, ...]
    marginal_response_gain: tuple[float, ...]
    sign_inversion_cell_ids: tuple[str, ...]
    decomposition_residual: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ModeInput:
    contract: str
    source: str
    probabilities: np.ndarray
    kernel_norms: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    probe_steps: int


def audit_history_estimands(request: HistoryEstimandRequest) -> HistoryEstimandAudit:
    targets = reconstruct_teacher_targets(request.path)
    metadata = load_teacher_rf_metadata(request.path)
    if metadata is None or metadata.context_gain_envelope is None:
        raise HistoryEstimandError("Complete adaptive teacher metadata is required")
    if (
        not 1 <= request.probe_steps <= targets.spike_counts.shape[2]
        or request.endogenous_trials < 1
    ):
        raise HistoryEstimandError("Probe and endogenous trial counts are invalid")
    pairs = _context_pairs(targets.source_ids, targets.context_ids)
    base_logits = _logit(targets.expected_probabilities)
    kernel_norms = _kernel_norms(metadata.static_kernel, metadata.context_gain_envelope)
    observed_history = history_trace(targets.spike_counts, decay=float(_SPIKE_HISTORY_DECAY))
    matched_history = _matched_history(observed_history, pairs)
    matched_probabilities = _sigmoid(
        base_logits[:, None] + float(_SPIKE_HISTORY_LOGIT_GAIN) * matched_history
    )
    standard_probabilities = _standard_train_rate_probabilities(
        base_logits,
        targets.spike_counts,
    )
    controlled_inputs: dict[RFHistoryContract, _ModeInput] = {
        "zero": _ModeInput(
            "zero",
            "all_zero",
            targets.expected_probabilities[:, None],
            kernel_norms,
            pairs,
            request.probe_steps,
        ),
        "matched_observed": _ModeInput(
            "matched_observed",
            "low_observed_history_reused_for_pair",
            matched_probabilities,
            kernel_norms,
            pairs,
            request.probe_steps,
        ),
        "standard_train_rate": _ModeInput(
            "standard_train_rate",
            "deterministic_train_rate_schedule",
            standard_probabilities,
            kernel_norms,
            pairs,
            request.probe_steps,
        ),
    }
    controlled = {
        contract: _mode_result(controlled_inputs[contract])
        for contract in RF_HISTORY_CONTRACTS
    }
    _, endogenous_probabilities = _sample_history_conditioned_spikes(
        np.random.default_rng(request.seed),
        base_logits,
        request.endogenous_trials,
    )
    modes = tuple(
        _mode_result(values)
        for values in (
            controlled_inputs["zero"],
            controlled_inputs["matched_observed"],
            controlled_inputs["standard_train_rate"],
            _ModeInput(
                "observed_history",
                "condition_specific_recorded_history",
                targets.conditional_probabilities,
                kernel_norms,
                pairs,
                request.probe_steps,
            ),
            _ModeInput(
                "endogenous_history",
                "teacher_generated_history_distribution",
                endogenous_probabilities,
                kernel_norms,
                pairs,
                request.probe_steps,
            ),
        )
    )
    direct = np.asarray(modes[0].response_gain, dtype=np.float64)
    marginal = np.asarray(modes[-1].response_gain, dtype=np.float64)
    mediated = marginal - direct
    inversion = np.sign(direct) != np.sign(marginal)
    residual = marginal - (direct + mediated)
    return HistoryEstimandAudit(
        request.probe_steps,
        request.endogenous_trials,
        targets.cell_ids,
        controlled,
        modes,
        tuple(float(value) for value in direct),
        tuple(float(value) for value in mediated),
        tuple(float(value) for value in marginal),
        tuple(cell for cell, changed in zip(targets.cell_ids, inversion, strict=True) if changed),
        tuple(float(value) for value in residual),
    )


def _mode_result(values: _ModeInput) -> HistoryModeResult:
    logit_gains = []
    probability_gains = []
    response_gains = []
    for low, high in values.pairs:
        low_probability = values.probabilities[low]
        high_probability = values.probabilities[high]
        low_slope = low_probability[:, -1] * (1 - low_probability[:, -1])
        high_slope = high_probability[:, -1] * (1 - high_probability[:, -1])
        low_norm = values.kernel_norms[low, -1]
        high_norm = values.kernel_norms[high, -1]
        logit_gains.append(np.log((high_norm + 1e-8) / (low_norm + 1e-8)))
        probability_gains.append(
            np.log(
                (high_slope * high_norm + 1e-8)
                / (low_slope * low_norm + 1e-8)
            ).mean(axis=0)
        )
        response_gains.append(
            high_probability[:, -values.probe_steps :].mean(axis=(0, 1))
            - low_probability[:, -values.probe_steps :].mean(axis=(0, 1))
        )
    return HistoryModeResult(
        values.contract,
        values.source,
        tuple(float(value) for value in np.stack(logit_gains).mean(axis=0)),
        tuple(float(value) for value in np.stack(probability_gains).mean(axis=0)),
        tuple(float(value) for value in np.stack(response_gains).mean(axis=0)),
    )


def _kernel_norms(static_kernel: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    cell_norms = np.linalg.norm(static_kernel.reshape(static_kernel.shape[0], -1), axis=1)
    if envelope.shape[2] != cell_norms.shape[0]:
        raise HistoryEstimandError("Teacher envelope and kernels disagree")
    return envelope * cell_norms[None, None]


def _matched_history(
    observed: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    matched = np.zeros_like(observed)
    for low, high in pairs:
        matched[low] = observed[low]
        matched[high] = observed[low]
    return matched


def _standard_train_rate_probabilities(
    base_logits: np.ndarray,
    spikes: np.ndarray,
) -> np.ndarray:
    split = ResponseSplit(
        cone_response=torch.zeros((spikes.shape[0], spikes.shape[2], 1)),
        spike_counts=torch.as_tensor(spikes),
        valid_mask=torch.ones_like(torch.as_tensor(spikes), dtype=torch.bool),
        source_ids=(),
        context_ids=(),
    )
    counts = standard_train_rate_history_counts(
        split,
        burn_in_steps=0,
        sequence_steps=spikes.shape[2],
    )
    standard_trace = history_trace(
        counts.numpy()[None],
        decay=float(_SPIKE_HISTORY_DECAY),
    )
    return _sigmoid(
        base_logits[:, None] + float(_SPIKE_HISTORY_LOGIT_GAIN) * standard_trace
    )


def _context_pairs(
    source_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    grouped: dict[str, dict[str, int]] = {}
    for index, (source, context) in enumerate(zip(source_ids, context_ids, strict=True)):
        grouped.setdefault(source, {})[context] = index
    pairs = tuple(
        (contexts["low"], contexts["high"])
        for contexts in grouped.values()
        if "low" in contexts and "high" in contexts
    )
    if not pairs:
        raise HistoryEstimandError("Matched low/high contexts are required")
    return pairs


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-np.clip(values, -20.0, 20.0)))).astype(np.float32)


__all__ = [
    "HistoryEstimandAudit",
    "HistoryEstimandError",
    "HistoryEstimandRequest",
    "HistoryModeResult",
    "audit_history_estimands",
]
