from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.point_process_teacher import (
    _SPIKE_HISTORY_DECAY,
    _SPIKE_HISTORY_LOGIT_GAIN,
    _sample_history_conditioned_spikes,
)
from data.rgc_response import load_rgc_response
from data.synthetic_teacher import (
    load_teacher_input_normalization,
    load_teacher_rf_metadata,
)
from evaluation.teacher_identifiability import reconstruct_teacher_targets
from evaluation.teacher_self_fit_model import (
    TeacherSelfFitBatch,
    TeacherSelfFitModel,
    TeacherSelfFitShape,
    context_recovery_basis,
    history_trace,
)
from evaluation.teacher_self_fit_metrics import (
    TeacherSelfFitPoint,
    TeacherSelfFitSummaryInput,
    signed_gains,
    summarize_teacher_self_fit,
)


class TeacherSelfFitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TeacherSelfFitRequest:
    path: str | Path
    trial_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    monte_carlo_seeds: int = 20
    max_iterations: int = 30
    probe_steps: int = 64
    seed: int = 0
    l2_penalty: float = 1e-4
    direction_gate: float = 0.90
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TeacherSelfFitAudit:
    monte_carlo_seeds: int
    max_iterations: int
    direction_gate: float
    teacher_context_gains: tuple[float, ...]
    true_history_gain: float
    minimum_passing_trial_count: int | None
    points: tuple[TeacherSelfFitPoint, ...]


@dataclass(frozen=True, slots=True)
class _LoadedTeacher:
    normalized_cones: np.ndarray
    context_basis: np.ndarray
    base_logits: np.ndarray
    teacher_low_kernel: np.ndarray
    teacher_high_kernel: np.ndarray
    cell_ids: tuple[str, ...]
    type_ids: tuple[str, ...]
    polarities: np.ndarray


@dataclass(frozen=True, slots=True)
class _FitPointInput:
    teacher: _LoadedTeacher
    settings: TeacherSelfFitRequest
    trial_count: int
    seed: int


def audit_teacher_self_fit(request: TeacherSelfFitRequest) -> TeacherSelfFitAudit:
    teacher = _load_teacher(request)
    _validate_request(request, teacher)
    points = tuple(
        _fit_point(_FitPointInput(teacher, request, count, request.seed + index * 100_000))
        for index, count in enumerate(request.trial_counts)
    )
    passing = tuple(
        point.trial_count
        for point in points
        if point.direction_recovery_rate >= request.direction_gate
        and point.context_gain_ci_supported_count == len(teacher.cell_ids)
    )
    teacher_gains = signed_gains(teacher.teacher_low_kernel, teacher.teacher_high_kernel)
    return TeacherSelfFitAudit(
        request.monte_carlo_seeds,
        request.max_iterations,
        request.direction_gate,
        tuple(float(value) for value in teacher_gains),
        float(_SPIKE_HISTORY_LOGIT_GAIN),
        min(passing) if passing else None,
        points,
    )


def _load_teacher(request: TeacherSelfFitRequest) -> _LoadedTeacher:
    session = load_rgc_response(request.path)
    normalization = load_teacher_input_normalization(
        request.path,
        session.cone_response.shape[2],
    )
    metadata = load_teacher_rf_metadata(request.path)
    targets = reconstruct_teacher_targets(request.path)
    if normalization is None or metadata is None:
        raise TeacherSelfFitError("Synthetic teacher normalization and kernels are required")
    probabilities = np.clip(targets.expected_probabilities, 1e-7, 1 - 1e-7)
    return _LoadedTeacher(
        normalization.normalize(session.cone_response).astype(np.float32),
        context_recovery_basis(session.context_ids, session.cone_response.shape[1]),
        np.log(probabilities / (1 - probabilities)).astype(np.float32),
        metadata.context_kernel_low,
        metadata.context_kernel_high,
        session.cells.ids,
        session.cells.type_ids,
        session.cells.polarities,
    )


def _validate_request(request: TeacherSelfFitRequest, teacher: _LoadedTeacher) -> None:
    if (
        not request.trial_counts
        or any(count < 2 for count in request.trial_counts)
        or request.monte_carlo_seeds < 1
        or request.max_iterations < 1
        or not 1 <= request.probe_steps <= teacher.base_logits.shape[1]
        or request.l2_penalty < 0
        or not 0 <= request.direction_gate <= 1
    ):
        raise TeacherSelfFitError("Teacher self-fit settings are invalid")


def _fit_point(values: _FitPointInput) -> TeacherSelfFitPoint:
    samples = tuple(
        _sample_history_conditioned_spikes(
            np.random.default_rng(values.seed + offset),
            values.teacher.base_logits,
            values.trial_count,
        )
        for offset in range(values.settings.monte_carlo_seeds)
    )
    spikes = np.stack([sample[0] for sample in samples])
    probabilities = np.stack([sample[1] for sample in samples])
    fit_count = values.trial_count // 2
    fit_spikes = spikes[:, :, :fit_count]
    heldout_spikes = spikes[:, :, fit_count:]
    heldout_probabilities = probabilities[:, :, fit_count:]
    device = torch.device(values.settings.device)
    model = TeacherSelfFitModel(
        TeacherSelfFitShape(
            values.settings.monte_carlo_seeds,
            spikes.shape[-1],
            values.teacher.teacher_low_kernel.shape[1],
            values.teacher.normalized_cones.shape[2],
        )
    ).to(device)
    with torch.no_grad():
        rates = torch.as_tensor(fit_spikes, device=device).mean(dim=(1, 2, 3))
        rates = rates.clamp(1e-4, 1 - 1e-4)
        model.bias.copy_(torch.logit(rates))
    fit_batch = _batch(values.teacher, fit_spikes, device)
    fit_targets = torch.as_tensor(fit_spikes, device=device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=values.settings.max_iterations,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        logits = model(fit_batch)
        nll = F.binary_cross_entropy_with_logits(logits, fit_targets)
        penalty = values.settings.l2_penalty * (
            model.base_kernel.square().mean() + model.context_kernel.square().mean()
        )
        loss = nll + penalty
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    heldout_batch = _batch(values.teacher, heldout_spikes, device)
    heldout_targets = torch.as_tensor(heldout_spikes, device=device)
    with torch.no_grad():
        heldout_logits = model(heldout_batch)
        heldout_nll = F.binary_cross_entropy_with_logits(
            heldout_logits,
            heldout_targets,
            reduction="none",
        ).mean(dim=(1, 2, 3, 4))
        teacher_oracle_nll = F.binary_cross_entropy(
            torch.as_tensor(heldout_probabilities, device=device),
            heldout_targets,
            reduction="none",
        ).mean(dim=(1, 2, 3, 4))
    recovery = float(values.teacher.context_basis[:, -1].max())
    low = model.base_kernel.detach().cpu().numpy()
    return summarize_teacher_self_fit(
        TeacherSelfFitSummaryInput(
            values.trial_count,
            fit_count,
            evaluations,
            low,
            low + recovery * model.context_kernel.detach().cpu().numpy(),
            model.context_kernel.detach().cpu().numpy(),
            values.teacher.teacher_low_kernel,
            values.teacher.teacher_high_kernel,
            heldout_nll.detach().cpu().numpy(),
            teacher_oracle_nll.detach().cpu().numpy(),
            model.history_gain.detach().cpu().numpy(),
            float(_SPIKE_HISTORY_LOGIT_GAIN),
            values.teacher.type_ids,
            values.teacher.polarities,
        )
    )


def _batch(
    teacher: _LoadedTeacher,
    spikes: np.ndarray,
    device: torch.device,
) -> TeacherSelfFitBatch:
    return TeacherSelfFitBatch(
        torch.as_tensor(teacher.normalized_cones, device=device),
        torch.as_tensor(teacher.context_basis, device=device),
        torch.as_tensor(
            history_trace(spikes, decay=float(_SPIKE_HISTORY_DECAY)),
            device=device,
        ),
    )


__all__ = [
    "TeacherSelfFitAudit",
    "TeacherSelfFitError",
    "TeacherSelfFitPoint",
    "TeacherSelfFitRequest",
    "audit_teacher_self_fit",
]
