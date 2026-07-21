from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, assert_never


class FeasibilityDecision(StrEnum):
    GO = "go"
    RUNS_WITHOUT_SUPPORT = "runs_without_support"
    NO_GO = "no_go"


EvidenceClass = Literal["formal_candidate", "non_formal_smoke"]


@dataclass(frozen=True, slots=True)
class FeasibilityEvidence:
    evidence_class: EvidenceClass
    structural_pass: bool
    dynamics_pass: bool
    skill_current: float
    rf_pass: bool
    humret_pass: bool
    parameters_clear_of_bounds: bool


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    decision: FeasibilityDecision
    failed_gates: tuple[str, ...]
    task_pass: bool
    rf_pass: bool
    humret_pass: bool


def assess_feasibility(evidence: FeasibilityEvidence) -> FeasibilityReport:
    task_pass = math.isfinite(evidence.skill_current) and evidence.skill_current > 0
    failed = []
    if not math.isfinite(evidence.skill_current):
        failed.append("non_finite_evidence")
    if not evidence.structural_pass:
        failed.append("structural")
    if not evidence.dynamics_pass:
        failed.append("dynamics")
    if not task_pass and math.isfinite(evidence.skill_current):
        failed.append("task_best_baseline")
    if not evidence.rf_pass:
        failed.append("rf")
    if not evidence.humret_pass:
        failed.append("humret")
    if not evidence.parameters_clear_of_bounds:
        failed.append("parameter_boundaries")

    match evidence.evidence_class:
        case "formal_candidate":
            decision = FeasibilityDecision.NO_GO if failed else FeasibilityDecision.GO
        case "non_formal_smoke":
            failed.append("non_formal_smoke")
            decision = FeasibilityDecision.RUNS_WITHOUT_SUPPORT
        case unreachable:
            assert_never(unreachable)
    return FeasibilityReport(
        decision=decision,
        failed_gates=tuple(failed),
        task_pass=task_pass,
        rf_pass=evidence.rf_pass,
        humret_pass=evidence.humret_pass,
    )
