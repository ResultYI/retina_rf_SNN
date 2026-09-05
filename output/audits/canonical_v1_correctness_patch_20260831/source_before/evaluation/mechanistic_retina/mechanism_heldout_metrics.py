from __future__ import annotations

from dataclasses import dataclass
import math
from typing import assert_never

import torch
from torch.nn import functional as F

from evaluation.mechanistic_retina.mechanism_checkpoints import tensors_sha256
from evaluation.mechanistic_retina.mechanism_heldout import (
    HeldoutPathway,
    HeldoutProbe,
    heldout_probes,
)
from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismTeacher,
    pathway_rfs,
)
from evaluation.mechanistic_retina.mechanism_run_types import AblationName
from evaluation.mechanistic_retina.mechanism_runs import ablation_clamps
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    expected_ce: float
    logit_rmse: float
    response_correlation: float


@dataclass(frozen=True, slots=True)
class DiagnosticResponse:
    name: str
    preregistered_name: str
    teacher: float
    full: float
    structural: float
    clamped: float
    direction_cosine: float


@dataclass(frozen=True, slots=True)
class PathwayMetrics:
    activation: float
    current: float
    sensitivity: float
    rf_norm: float
    rf_cosine: float
    rf_sha256: str
    teacher_component_norm: float
    teacher_component_sha256: str


@dataclass(frozen=True, slots=True)
class HeldoutSeedMetrics:
    pathway_name: HeldoutPathway
    seed: int
    optimizer_steps: int
    full: PredictionMetrics
    structural: PredictionMetrics
    clamped: PredictionMetrics
    structural_ce_delta: float
    clamp_ce_delta: float
    response_direction_consistent: bool
    responses: tuple[DiagnosticResponse, ...]
    pathway: PathwayMetrics


@dataclass(frozen=True, slots=True)
class HeldoutEvaluationRequest:
    pathway: HeldoutPathway
    teacher: MechanismTeacher
    full_model: MechanisticGraphTemporalRetina
    structural_model: MechanisticGraphTemporalRetina
    data: MechanisticSeedData
    structural_variant: AblationName
    seed: int


@dataclass(frozen=True, slots=True)
class HeldoutEvaluationError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def evaluate_heldout(request: HeldoutEvaluationRequest) -> HeldoutSeedMetrics:
    clamps = _pathway_clamps(request.pathway)
    if ablation_clamps(request.structural_variant) != clamps:
        raise HeldoutEvaluationError("structural variant does not match held-out pathway")
    probes = heldout_probes(request.teacher.model, request.data, request.pathway)
    teacher_logits = _probe_logits(request.teacher.model, probes, frozenset())
    teacher_clamped = _probe_logits(request.teacher.model, probes, clamps)
    full_logits = _probe_logits(request.full_model, probes, frozenset())
    structural_logits = _probe_logits(request.structural_model, probes, clamps)
    clamped_logits = _probe_logits(request.full_model, probes, clamps)
    target = torch.sigmoid(teacher_logits)
    full = _prediction(full_logits, teacher_logits, target)
    structural = _prediction(structural_logits, teacher_logits, target)
    clamped = _prediction(clamped_logits, teacher_logits, target)
    responses = _responses(
        probes,
        teacher_logits,
        teacher_clamped,
        full_logits,
        structural_logits,
        clamped_logits,
    )
    return HeldoutSeedMetrics(
        request.pathway,
        request.seed,
        0,
        full,
        structural,
        clamped,
        structural.expected_ce - full.expected_ce,
        clamped.expected_ce - full.expected_ce,
        all(response.direction_cosine > 0.0 for response in responses),
        responses,
        _pathway_metrics(request, probes),
    )


def _probe_logits(
    model: MechanisticGraphTemporalRetina,
    probes: tuple[HeldoutProbe, ...],
    clamps: frozenset[PathwayClamp],
) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            tuple(
                model.forward_sequence(
                    probe.stimulus,
                    observed_counts=probe.history,
                    clamps=clamps,
                ).logits
                for probe in probes
            ),
            dim=0,
        )


def _prediction(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
) -> PredictionMetrics:
    mask = torch.ones_like(target)
    ce = expected_bernoulli_nll(logits, target, mask)
    rmse = torch.sqrt((logits - teacher_logits).square().mean())
    return PredictionMetrics(float(ce), float(rmse), _correlation(logits, teacher_logits))


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.flatten().double()
    right = second.flatten().double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= torch.finfo(left.dtype).eps:
        return float(torch.equal(first, second))
    return float(torch.dot(left, right) / denominator)


def _responses(
    probes: tuple[HeldoutProbe, ...],
    teacher: torch.Tensor,
    teacher_clamped: torch.Tensor,
    full: torch.Tensor,
    structural: torch.Tensor,
    clamped: torch.Tensor,
) -> tuple[DiagnosticResponse, ...]:
    rows = []
    for index, probe in enumerate(probes):
        teacher_probability = torch.sigmoid(teacher[index, :, probe.cell])
        teacher_off = torch.sigmoid(teacher_clamped[index, :, probe.cell])
        full_probability = torch.sigmoid(full[index, :, probe.cell])
        full_off = torch.sigmoid(clamped[index, :, probe.cell])
        rows.append(
            DiagnosticResponse(
                probe.name,
                probe.preregistered_name,
                float(teacher_probability.mean()),
                float(full_probability.mean()),
                float(torch.sigmoid(structural[index, :, probe.cell]).mean()),
                float(full_off.mean()),
                _direction_cosine(
                    teacher_probability - teacher_off,
                    full_probability - full_off,
                ),
            )
        )
    return tuple(rows)


def _direction_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(first.flatten().double(), second.flatten().double(), dim=0)
    )


def _pathway_metrics(
    request: HeldoutEvaluationRequest,
    probes: tuple[HeldoutProbe, ...],
) -> PathwayMetrics:
    activation, current, sensitivity = _state_metrics(
        request.full_model, probes, request.pathway
    )
    history = torch.zeros_like(request.teacher.validation_probability[:2])
    rf = pathway_rfs(
        request.full_model,
        request.data.validation_cones[:2],
        history,
    )[request.pathway.value]
    teacher = request.teacher.pathway_rf
    cosine = F.cosine_similarity(rf.flatten().double(), teacher.flatten().double(), dim=0)
    return PathwayMetrics(
        activation,
        current,
        sensitivity,
        float(rf.norm()),
        float(cosine),
        tensors_sha256({"pathway_rf": rf}),
        float(teacher.norm()),
        tensors_sha256({"teacher_component": teacher}),
    )


def _state_metrics(
    model: MechanisticGraphTemporalRetina,
    probes: tuple[HeldoutProbe, ...],
    pathway: HeldoutPathway,
) -> tuple[float, float, float]:
    values = tuple(_one_state_metric(model, probe, pathway) for probe in probes)
    return tuple(float(sum(row[index] for row in values) / len(values)) for index in range(3))


def _one_state_metric(
    model: MechanisticGraphTemporalRetina,
    probe: HeldoutProbe,
    pathway: HeldoutPathway,
) -> tuple[float, float, float]:
    output = model.forward_sequence(probe.stimulus, observed_counts=probe.history)
    match pathway:
        case HeldoutPathway.H1:
            currents = (output.h1_surround_contribution,)
            activation = output.h1_state.abs().mean()
        case HeldoutPathway.AC:
            currents = (output.amacrine_local_current, output.amacrine_transient_current)
            activation = 0.5 * (
                output.amacrine_local_state.abs().mean()
                + output.amacrine_transient_state.abs().mean()
            )
        case unreachable:
            assert_never(unreachable)
    gradients = torch.autograd.grad(output.logits.sum(), currents)
    current = sum(value.abs().mean() for value in currents)
    sensitivity = sum(value.abs().mean() for value in gradients)
    values = (float(activation.detach()), float(current.detach()), float(sensitivity.detach()))
    if not all(math.isfinite(value) for value in values):
        raise HeldoutEvaluationError("held-out pathway state is non-finite")
    return values


def _pathway_clamps(pathway: HeldoutPathway) -> frozenset[PathwayClamp]:
    match pathway:
        case HeldoutPathway.H1:
            return frozenset({PathwayClamp.H1})
        case HeldoutPathway.AC:
            return frozenset(
                {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
            )
        case unreachable:
            assert_never(unreachable)


__all__ = ["DiagnosticResponse", "HeldoutEvaluationError", "HeldoutEvaluationRequest", "HeldoutSeedMetrics", "PathwayMetrics", "PredictionMetrics", "evaluate_heldout"]
