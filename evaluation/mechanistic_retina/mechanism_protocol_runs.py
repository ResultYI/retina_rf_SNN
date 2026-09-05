from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismRunConfig,
    MechanismTeacher,
)
from evaluation.mechanistic_retina.mechanism_run_data import SampledCondition
from evaluation.mechanistic_retina.mechanism_run_types import (
    MechanismRunEvidence,
    ProgressEvent,
    TeacherRunRequest,
)
from evaluation.mechanistic_retina.mechanism_runs import run_teacher_seed
from evaluation.mechanistic_retina.mechanism_scoring import MechanismScore, final_case
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
)
from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class PhaseRequest:
    teachers: tuple[MechanismTeacher, ...]
    data: MechanisticSeedData
    candidate: Candidate0Reference
    config: MechanismRunConfig
    sampled: SampledCondition | None
    progress: Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class SampledRequest:
    phase: PhaseRequest
    noise_score: MechanismScore


def run_phase(request: PhaseRequest) -> tuple[MechanismRunEvidence, ...]:
    rows = []
    for seed in request.config.seeds:
        for teacher in request.teachers:
            rows.extend(
                run_teacher_seed(
                    TeacherRunRequest(
                        teacher,
                        request.data,
                        request.candidate,
                        request.config,
                        seed,
                        request.sampled,
                        request.progress,
                    )
                )
            )
    return tuple(rows)


def sampled_if_supported(
    request: SampledRequest,
) -> tuple[tuple[MechanismRunEvidence, ...], list[JsonValue]]:
    if final_case(request.noise_score, None) != "MECHANISM-IDENTIFIABLE-RETINA-SUPPORTED":
        return (), []
    rows = []
    banks: list[JsonValue] = []
    phase = request.phase
    for index, teacher in enumerate(phase.teachers):
        bank = slice_spike_bank(
            generate_nested_spike_bank(
                teacher.train_probability,
                teacher.validation_probability,
                seed=phase.config.bank_seed + index,
                max_trials=2,
            ),
            2,
        )
        banks.append(
            {
                "teacher": teacher.name.value,
                "seed": phase.config.bank_seed + index,
                "train_sha256": bank.train_sha256,
                "validation_sha256": bank.validation_sha256,
            }
        )
        rows.extend(
            run_phase(
                PhaseRequest(
                    (teacher,),
                    phase.data,
                    phase.candidate,
                    phase.config,
                    SampledCondition(bank.train_spikes, bank.validation_spikes),
                    phase.progress,
                )
            )
        )
    return tuple(rows), banks


__all__ = ["PhaseRequest", "SampledRequest", "run_phase", "sampled_if_supported"]
