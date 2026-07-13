from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class FeasibilityDecision(StrEnum):
    GO = "go"
    RUNS_WITHOUT_SUPPORT = "runs_without_support"
    NO_GO = "no_go"


@dataclass(frozen=True, slots=True)
class FeasibilityEvidence:
    structural_pass: bool
    dynamics_pass: bool
    fine_skill: float
    coarse_skill: float
    trained_core_skill: float
    residual_gain_fraction: float
    rf_agreement_fraction: float
    parameters_clear_of_bounds: bool
    functional_pass: bool = False


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    decision: FeasibilityDecision
    failed_gates: tuple[str, ...]


def assess_feasibility(evidence: FeasibilityEvidence) -> FeasibilityReport:
    failed = []
    numeric = (
        evidence.fine_skill,
        evidence.coarse_skill,
        evidence.trained_core_skill,
        evidence.residual_gain_fraction,
        evidence.rf_agreement_fraction,
    )
    if not all(math.isfinite(value) for value in numeric):
        return FeasibilityReport(
            FeasibilityDecision.NO_GO,
            ("non_finite_evidence",),
        )
    if not (
        0 <= evidence.residual_gain_fraction <= 1
        and 0 <= evidence.rf_agreement_fraction <= 1
    ):
        return FeasibilityReport(
            FeasibilityDecision.NO_GO,
            ("invalid_fraction_evidence",),
        )
    if not evidence.structural_pass:
        failed.append("structural")
    if not evidence.dynamics_pass:
        failed.append("dynamics")
    if failed:
        return FeasibilityReport(FeasibilityDecision.NO_GO, tuple(failed))
    if evidence.fine_skill <= 0:
        failed.append("fine_best_baseline")
    if evidence.coarse_skill <= 0:
        failed.append("coarse_best_baseline")
    if evidence.trained_core_skill <= 0:
        failed.append("trained_core")
    if failed:
        return FeasibilityReport(FeasibilityDecision.NO_GO, tuple(failed))
    if evidence.fine_skill < 0.05:
        failed.append("fine_practical_prediction_effect")
    if evidence.coarse_skill < 0.05:
        failed.append("coarse_practical_prediction_effect")
    if evidence.residual_gain_fraction > 0.25:
        failed.append("residual_dominance")
    if evidence.rf_agreement_fraction < 0.80:
        failed.append("rf_agreement")
    if not evidence.functional_pass:
        failed.append("human_functional_agreement")
    if not evidence.parameters_clear_of_bounds:
        failed.append("parameter_boundaries")
    decision = (
        FeasibilityDecision.GO
        if not failed
        else FeasibilityDecision.RUNS_WITHOUT_SUPPORT
    )
    return FeasibilityReport(decision, tuple(failed))
