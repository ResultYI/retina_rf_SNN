from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import assert_never

import torch

from evaluation.mechanistic_retina.direct_metrics import rf_payload, rf_summary
from evaluation.mechanistic_retina.direct_model_eval import build_model
from evaluation.mechanistic_retina.h1_teachers import (
    H1TeacherRequest,
    build_matched_h1_teachers,
)
from evaluation.mechanistic_retina.mechanism_runtime import MechanismRunConfig
from evaluation.mechanistic_retina.mechanism_diagnosis_support import (
    bias_ce,
    legacy_closure,
)
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.structural_ablation import (
    NoiseFreeTrainingRequest,
    train_noise_free,
)
from evaluation.mechanistic_retina.subspace_overlap import (
    SubspaceOverlapRequest,
    SubspaceOverlapResult,
    fisher_subspace_overlap,
)
from models.mechanistic_retina.contracts import PathwayClamp
from training.mechanistic_retina.stages import MechanisticSeedData


@unique
class LegacyAblationName(StrEnum):
    FULL = "full"
    NO_H1 = "no-H1"
    NO_AC = "no-AC"


@dataclass(frozen=True, slots=True)
class LegacyDiagnosisRequest:
    data: MechanisticSeedData
    candidate: Candidate0Reference
    config: MechanismRunConfig
    progress: Callable[[str, int, float], None]


@dataclass(frozen=True, slots=True)
class LegacyAblation:
    name: LegacyAblationName
    validation_ce: float
    bias_ce: float
    replaceability_ratio: float
    rf: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LegacyDiagnosis:
    exact_closure: Mapping[str, bool | float]
    ablations: tuple[LegacyAblation, ...]
    overlap: SubspaceOverlapResult
    decision: str


@dataclass(frozen=True, slots=True)
class LegacyAblationRequest:
    diagnosis: LegacyDiagnosisRequest
    name: LegacyAblationName
    train_probability: torch.Tensor
    validation_probability: torch.Tensor
    bias_ce: float


def run_legacy_diagnosis(request: LegacyDiagnosisRequest) -> LegacyDiagnosis:
    teacher_model = build_model(request.data, 19)
    teachers = build_matched_h1_teachers(
        H1TeacherRequest(
            teacher_model,
            request.data.train_cones,
            request.data.validation_cones,
            request.candidate.rf,
            request.data.train_mask[:, 0],
            request.data.validation_mask[:, 0],
            -2.0,
        )
    )
    baseline_ce = bias_ce(
        teachers.present_train_probability,
        teachers.present_validation_probability,
        request.data,
    )
    runs = tuple(
        _legacy_ablation(
            LegacyAblationRequest(
                request,
                name,
                teachers.present_train_probability,
                teachers.present_validation_probability,
                baseline_ce,
            )
        )
        for name in (
            LegacyAblationName.FULL,
            LegacyAblationName.NO_H1,
            LegacyAblationName.NO_AC,
        )
    )
    full_ce = runs[0].validation_ce
    normalized = tuple(
        LegacyAblation(
            run.name,
            run.validation_ce,
            run.bias_ce,
            (baseline_ce - run.validation_ce) / max(1e-12, baseline_ce - full_ce),
            run.rf,
        )
        for run in runs
    )
    overlap_model = build_model(request.data, 19)
    overlap = fisher_subspace_overlap(
        SubspaceOverlapRequest(
            overlap_model,
            request.data.validation_cones[:1, :64],
            torch.zeros_like(request.data.validation_probability[:1, 0, :64]),
            16,
        )
    )
    closure = legacy_closure(overlap_model, request.data)
    structural = (
        not bool(closure["parameterizable_absence"])
        or overlap.h1_unique_fraction < 0.20
        or overlap.ac_unique_fraction < 0.20
        or any(run.replaceability_ratio >= 0.90 for run in normalized[1:])
    )
    return LegacyDiagnosis(
        closure,
        normalized,
        overlap,
        "structural-change" if structural else "light-change",
    )


def _legacy_ablation(
    request: LegacyAblationRequest,
) -> LegacyAblation:
    diagnosis = request.diagnosis
    model = build_model(diagnosis.data, 19)
    clamps = _clamps(request.name)
    training = train_noise_free(
        NoiseFreeTrainingRequest(
            model,
            diagnosis.data.train_cones,
            torch.zeros_like(request.train_probability),
            request.train_probability,
            diagnosis.data.train_mask[:, 0],
            diagnosis.data.validation_cones,
            torch.zeros_like(request.validation_probability),
            request.validation_probability,
            diagnosis.data.validation_mask[:, 0],
            clamps,
            diagnosis.config.steps,
            diagnosis.config.checkpoints,
            diagnosis.config.learning_rate,
            diagnosis.config.batch_size,
            19,
            lambda step, ce: diagnosis.progress(request.name.value, step, ce),
        )
    )
    ce = training.checkpoints[-1].validation_ce
    learned = effective_rf(
        model,
        diagnosis.data.validation_cones[:2],
        torch.zeros_like(request.validation_probability[:2]),
        clamps=clamps,
    )
    summary = rf_summary(
        learned,
        diagnosis.candidate.rf,
        diagnosis.data.cone_positions,
        diagnosis.data.cell_positions,
        diagnosis.candidate.metadata,
        pair_count=1,
    )
    return LegacyAblation(
        request.name,
        ce,
        request.bias_ce,
        0.0,
        dict(rf_payload(summary)),
    )


def _clamps(name: LegacyAblationName) -> frozenset[PathwayClamp]:
    match name:
        case LegacyAblationName.FULL:
            return frozenset()
        case LegacyAblationName.NO_H1:
            return frozenset({PathwayClamp.H1})
        case LegacyAblationName.NO_AC:
            return frozenset(
                {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
            )
        case unreachable:
            assert_never(unreachable)


__all__ = ["LegacyDiagnosis", "LegacyDiagnosisRequest", "run_legacy_diagnosis"]
