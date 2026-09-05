from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique

from evaluation.mechanistic_retina.direct_metrics import DirectRFSummary
from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismRunConfig,
    MechanismTeacher,
)
from evaluation.mechanistic_retina.mechanism_run_data import SampledCondition
from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.mechanistic_retina.structural_ablation import NoiseFreeTrainingResult
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.stages import MechanisticSeedData


@unique
class AblationName(StrEnum):
    FULL = "full"
    NO_H1 = "no-H1"
    NO_AC = "no-AC"
    BC_ONLY = "BC-only"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: str
    teacher: str
    model: str
    seed: int
    step: int
    ce: float
    rf: float
    gate: float


@dataclass(frozen=True, slots=True)
class TeacherRunRequest:
    teacher: MechanismTeacher
    data: MechanisticSeedData
    candidate: Candidate0Reference
    config: MechanismRunConfig
    seed: int
    sampled: SampledCondition | None
    progress: Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    h1: float
    ac_local: float
    ac_transient: float
    history: float


@dataclass(frozen=True, slots=True)
class MechanismRunEvidence:
    phase: str
    teacher: str
    ablation: AblationName
    seed: int
    model: MechanisticGraphTemporalRetina
    training: NoiseFreeTrainingResult
    validation_ce: float
    bias_ce: float
    no_h1_ce: float
    no_ac_ce: float
    rf: DirectRFSummary
    gates: GateSnapshot
    pathway_norms: Mapping[str, float]
    pathway_cosines: Mapping[str, float]
    pathway_sum_error: float


__all__ = [
    "AblationName",
    "GateSnapshot",
    "MechanismRunEvidence",
    "ProgressEvent",
    "TeacherRunRequest",
]
