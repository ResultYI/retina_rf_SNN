from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.gain_audit import (
    CellSpecificGainAudit,
    PathwayGainAudit,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters
from training.mechanistic_retina.real_sampled import (
    RealSpikeTrainingError,
    SpikePredictionMetrics,
    _validation_metrics,
)


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    max_steps: int
    evaluation_interval: int
    patience: int
    min_delta: float


@dataclass(frozen=True, slots=True)
class EarlyStoppingTrainingRequest:
    model: MechanisticGraphTemporalRetina
    train: RealSequenceSplit
    validation: RealSequenceSplit
    learning_rate: float
    batch_size: int
    seed: int
    stopping: EarlyStoppingConfig


@dataclass(frozen=True, slots=True)
class ValidationTracePoint:
    step: int
    population_nll: float


@dataclass(frozen=True, slots=True)
class EarlyStoppingTrainingResult:
    raw_metrics: SpikePredictionMetrics
    best_metrics: SpikePredictionMetrics
    best_step: int
    completed_steps: int
    stopped_early: bool
    validation_trace: tuple[ValidationTracePoint, ...]
    gradients_finite: bool
    actually_updated: tuple[str, ...]
    nonself_connection_gradient_nonzero: bool
    nonself_connection_optimizer_updated: bool
    nonself_connection_updated: bool
    nonself_connection_max_abs_gradient: float
    nonself_connection_update_norm: float
    cell_gain_audit: CellSpecificGainAudit | None


def fit_real_spike_model_early_stopping(
    request: EarlyStoppingTrainingRequest,
) -> EarlyStoppingTrainingResult:
    _validate(request)
    torch.manual_seed(request.seed)
    optimizer = build_phase1_optimizer(request.model, learning_rate=request.learning_rate)
    parameters = phase1_parameters(request.model)
    names = {id(parameter): name for name, parameter in request.model.named_parameters()}
    initial = tuple(parameter.detach().clone() for parameter in parameters)
    connection_initial = request.model.shared_subunits.raw_connections.detach().clone()
    nonself = request.model.shared_subunits.edge_index[0] != request.model.shared_subunits.edge_index[1]
    gains = request.model.cell_gains
    gain_initial = (
        None
        if gains is None
        else tuple(raw.detach().clone() for raw in gains.raw_parameters)
    )
    gain_gradient_peaks = (
        None
        if gains is None
        else tuple(torch.zeros_like(raw) for raw in gains.raw_parameters)
    )
    raw_metrics = _validation_metrics(request.model, request.validation)
    best_metrics = raw_metrics
    best_state = deepcopy(request.model.state_dict())
    best_step = 0
    plateau_reference = raw_metrics.population_nll
    trace = [ValidationTracePoint(0, raw_metrics.population_nll)]
    stale_evaluations = 0
    gradients_finite = True
    max_nonself_gradient = 0.0
    nonself_optimizer_updated = False
    generator = torch.Generator().manual_seed(request.seed + 1_000_003)
    completed_steps = 0
    stopped_early = False
    request.model.train()
    for step in range(1, request.stopping.max_steps + 1):
        indices = torch.randint(
            request.train.cone_drive.shape[0],
            (request.batch_size,),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = request.model.forward_sequence(
            request.train.cone_drive[indices],
            observed_counts=request.train.spike_events[indices],
        ).logits
        loss = expected_bernoulli_nll(
            logits,
            request.train.spike_events[indices],
            request.train.valid_mask[indices],
        )
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        connection_gradient = request.model.shared_subunits.raw_connections.grad
        if gains is not None and gain_gradient_peaks is not None:
            for raw, peak in zip(
                gains.raw_parameters, gain_gradient_peaks, strict=True
            ):
                if raw.grad is not None:
                    peak.copy_(torch.maximum(peak, raw.grad.detach().abs()))
        if connection_gradient is not None and bool(nonself.any()):
            max_nonself_gradient = max(
                max_nonself_gradient,
                float(connection_gradient[nonself].abs().max()),
            )
        nonself_before_step = (
            request.model.shared_subunits.raw_connections.detach()[nonself].clone()
        )
        optimizer.step()
        nonself_optimizer_updated = nonself_optimizer_updated or not torch.equal(
            nonself_before_step,
            request.model.shared_subunits.raw_connections.detach()[nonself],
        )
        request.model.project_mechanism_parameters()
        completed_steps = step
        if step % request.stopping.evaluation_interval and step < request.stopping.max_steps:
            continue
        metrics = _validation_metrics(request.model, request.validation)
        trace.append(ValidationTracePoint(step, metrics.population_nll))
        if metrics.population_nll < best_metrics.population_nll:
            best_metrics = metrics
            best_state = deepcopy(request.model.state_dict())
            best_step = step
        if metrics.population_nll < plateau_reference - request.stopping.min_delta:
            plateau_reference = metrics.population_nll
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if stale_evaluations >= request.stopping.patience:
            stopped_early = True
            break
        request.model.train()
    request.model.load_state_dict(best_state)
    updated = tuple(
        names[id(parameter)]
        for before, parameter in zip(initial, parameters, strict=True)
        if not torch.equal(before, parameter.detach())
    )
    connection_change = (
        request.model.shared_subunits.raw_connections.detach() - connection_initial
    )
    nonself_update_norm = float(torch.linalg.vector_norm(connection_change[nonself]))
    gain_audit = None
    if gains is not None and gain_initial is not None and gain_gradient_peaks is not None:
        gain_audit = CellSpecificGainAudit(
            pathways=tuple(
                PathwayGainAudit(
                    name=name,
                    all_gradient_nonzero=bool((peak > 0).all()),
                    all_best_updated=bool((raw.detach() != before).all()),
                    min_peak_abs_gradient=float(peak.min()),
                    max_peak_abs_gradient=float(peak.max()),
                    best_update_norm=float(
                        torch.linalg.vector_norm(raw.detach() - before)
                    ),
                    best=tuple(float(value) for value in best),
                )
                for name, before, peak, raw, best in zip(
                    gains.pathway_names,
                    gain_initial,
                    gain_gradient_peaks,
                    gains.raw_parameters,
                    gains.audit_values.detach().unbind(dim=1),
                    strict=True,
                )
            )
        )
    return EarlyStoppingTrainingResult(
        raw_metrics=raw_metrics,
        best_metrics=best_metrics,
        best_step=best_step,
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        validation_trace=tuple(trace),
        gradients_finite=gradients_finite,
        actually_updated=updated,
        nonself_connection_gradient_nonzero=max_nonself_gradient > 0.0,
        nonself_connection_optimizer_updated=nonself_optimizer_updated,
        nonself_connection_updated=nonself_update_norm > 0.0,
        nonself_connection_max_abs_gradient=max_nonself_gradient,
        nonself_connection_update_norm=nonself_update_norm,
        cell_gain_audit=gain_audit,
    )


def _validate(request: EarlyStoppingTrainingRequest) -> None:
    stopping = request.stopping
    if (
        request.learning_rate <= 0
        or request.batch_size < 1
        or stopping.max_steps < 1
        or stopping.evaluation_interval < 1
        or stopping.patience < 1
        or stopping.min_delta < 0
    ):
        raise RealSpikeTrainingError("early-stopping training values are invalid")
    for split in (request.train, request.validation):
        if split.cone_drive.shape[:2] != split.spike_events.shape[:2]:
            raise RealSpikeTrainingError("real stimulus and spike time axes differ")
        if split.spike_events.shape != split.valid_mask.shape:
            raise RealSpikeTrainingError("real spike targets and masks differ")


__all__ = [
    "CellSpecificGainAudit",
    "EarlyStoppingConfig",
    "EarlyStoppingTrainingRequest",
    "EarlyStoppingTrainingResult",
    "ValidationTracePoint",
    "fit_real_spike_model_early_stopping",
]
