from __future__ import annotations

from typing import assert_never

import torch
from torch.nn import functional as F

from evaluation.mechanistic_retina.direct_metrics import rf_summary
from evaluation.mechanistic_retina.mechanism_identifiability import build_student, pathway_rfs
from evaluation.mechanistic_retina.mechanism_run_data import (
    SampledCondition,
    TrainingArrayRequest,
    TrainingArrays,
    bias_ce,
    build_training_arrays,
)
from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    GateSnapshot,
    MechanismRunEvidence,
    ProgressEvent,
    TeacherRunRequest,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.structural_ablation import (
    NoiseFreeTrainingRequest,
    train_noise_free,
)
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll


def run_teacher_seed(
    request: TeacherRunRequest,
) -> tuple[MechanismRunEvidence, ...]:
    return tuple(
        run_teacher_ablation(request, ablation)
        for ablation in (
            AblationName.FULL,
            AblationName.NO_H1,
            AblationName.NO_AC,
            AblationName.BC_ONLY,
        )
    )


def run_teacher_ablation(
    request: TeacherRunRequest,
    ablation: AblationName,
) -> MechanismRunEvidence:
    model = build_student(request.data, request.seed)
    arrays = build_training_arrays(
        TrainingArrayRequest(request.teacher, request.data, request.sampled)
    )
    clamps = ablation_clamps(ablation)
    phase = "noise-free" if request.sampled is None else "sampled-T2"

    def progress(step: int, ce: float) -> None:
        gates = _gates(model)
        request.progress(
            ProgressEvent(
                phase,
                request.teacher.name.value,
                ablation.value,
                request.seed,
                step,
                ce,
                0.0,
                max(gates.h1, gates.ac_local, gates.ac_transient),
            )
        )

    training = train_noise_free(
        NoiseFreeTrainingRequest(
            model,
            arrays.train_cones,
            arrays.train_observed,
            arrays.train_target,
            arrays.train_mask,
            arrays.validation_cones,
            arrays.validation_observed,
            arrays.validation_target,
            arrays.validation_mask,
            clamps,
            request.config.steps,
            request.config.checkpoints,
            request.config.learning_rate,
            request.config.batch_size,
            request.seed,
            progress,
        )
    )
    ce = _ce(model, arrays, clamps)
    no_h1_ce = _ce(model, arrays, clamps | frozenset({PathwayClamp.H1}))
    no_ac_ce = _ce(
        model,
        arrays,
        clamps
        | frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
    )
    bias_only_ce = bias_ce(arrays)
    rf_cones = arrays.validation_cones[:2]
    rf_history = arrays.validation_observed[:2]
    learned_rf = effective_rf(model, rf_cones, rf_history)
    rf = rf_summary(
        learned_rf,
        request.teacher.total_rf,
        request.data.cone_positions,
        request.data.cell_positions,
        request.candidate.metadata,
        pair_count=1,
    )
    pathways = pathway_rfs(model, rf_cones, rf_history)
    pathway_sum = sum(pathways.values(), torch.zeros_like(learned_rf))
    teacher_path = request.teacher.pathway_rf.flatten().double()
    cosines = {
        name: float(F.cosine_similarity(value.flatten().double(), teacher_path, dim=0))
        for name, value in pathways.items()
    }
    return MechanismRunEvidence(
        phase,
        request.teacher.name.value,
        ablation,
        request.seed,
        model,
        training,
        ce,
        bias_only_ce,
        no_h1_ce,
        no_ac_ce,
        rf,
        _gates(model),
        {name: float(value.norm()) for name, value in pathways.items()},
        cosines,
        float((pathway_sum - learned_rf).abs().max()),
    )


def ablation_clamps(ablation: AblationName) -> frozenset[PathwayClamp]:
    ac = {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
    match ablation:
        case AblationName.FULL:
            return frozenset()
        case AblationName.NO_H1:
            return frozenset({PathwayClamp.H1})
        case AblationName.NO_AC:
            return frozenset(ac)
        case AblationName.BC_ONLY:
            return frozenset(ac | {PathwayClamp.H1})
        case unreachable:
            assert_never(unreachable)


def _ce(
    model: MechanisticGraphTemporalRetina,
    arrays: TrainingArrays,
    clamps: frozenset[PathwayClamp],
) -> float:
    with torch.no_grad():
        logits = model.forward_sequence(
            arrays.validation_cones,
            observed_counts=arrays.validation_observed,
            clamps=clamps,
        ).logits
        return float(expected_bernoulli_nll(logits, arrays.validation_target, arrays.validation_mask))


def _gates(model: MechanisticGraphTemporalRetina) -> GateSnapshot:
    return GateSnapshot(
        float(model.gates.h1.detach()),
        float(model.gates.ac_local.detach()),
        float(model.gates.ac_transient.detach()),
        float(model.gates.history.detach()),
    )


__all__ = [
    "AblationName",
    "GateSnapshot",
    "MechanismRunEvidence",
    "ProgressEvent",
    "SampledCondition",
    "TeacherRunRequest",
    "ablation_clamps",
    "run_teacher_ablation",
    "run_teacher_seed",
]
