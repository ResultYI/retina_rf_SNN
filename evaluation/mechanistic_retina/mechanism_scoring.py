from __future__ import annotations

from dataclasses import dataclass
import statistics

from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    MechanismRunEvidence,
)


@dataclass(frozen=True, slots=True)
class MechanismScore:
    h1_passing_seeds: int
    ac_passing_seeds: int
    base_h1_passing_seeds: int
    base_ac_passing_seeds: int
    rf_passing_seeds: int
    h1_passed: bool
    ac_passed: bool
    base_passed: bool
    rf_passed: bool


@dataclass(frozen=True, slots=True)
class MechanismScoreError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def score_runs(
    runs: tuple[MechanismRunEvidence, ...],
    *,
    sampled: bool,
) -> MechanismScore:
    by_teacher = _by_teacher(runs)
    h1_full = by_teacher["H1-specific"][AblationName.FULL]
    ac_full = by_teacher["AC-specific"][AblationName.FULL]
    base_full = by_teacher["Base"][AblationName.FULL]
    h1_retrained = by_teacher["H1-specific"][AblationName.NO_H1]
    ac_retrained = by_teacher["AC-specific"][AblationName.NO_AC]
    h1_passes = tuple(
        _present_h1(full, h1_retrained[index])
        for index, full in enumerate(h1_full)
    )
    ac_passes = tuple(
        _present_ac(full, ac_retrained[index])
        for index, full in enumerate(ac_full)
    )
    present_h1_norm = statistics.median(run.pathway_norms["H1"] for run in h1_full)
    present_ac_norm = statistics.median(run.pathway_norms["AC"] for run in ac_full)
    present_h1_loss = statistics.median(run.no_h1_ce - run.validation_ce for run in h1_full)
    present_ac_loss = statistics.median(run.no_ac_ce - run.validation_ce for run in ac_full)
    base_h1 = tuple(
        run.gates.h1 <= 0.02
        and run.pathway_norms["H1"] <= 0.25 * present_h1_norm
        and abs(run.no_h1_ce - run.validation_ce) <= 0.25 * present_h1_loss
        for run in base_full
    )
    base_ac = tuple(
        max(run.gates.ac_local, run.gates.ac_transient) <= 0.10
        and run.pathway_norms["AC"] <= 0.25 * present_ac_norm
        and abs(run.no_ac_ce - run.validation_ce) <= 0.25 * present_ac_loss
        for run in base_full
    )
    global_gate = 0.80 if sampled else 0.85
    rf_passes = tuple(_rf_pass(run, global_gate) for run in h1_full + ac_full + base_full)
    return MechanismScore(
        sum(h1_passes),
        sum(ac_passes),
        sum(base_h1),
        sum(base_ac),
        sum(rf_passes),
        sum(h1_passes) >= 2,
        sum(ac_passes) >= 2,
        sum(base_h1) >= 2 and sum(base_ac) >= 2,
        sum(rf_passes) >= 6,
    )


def final_case(
    noise_free: MechanismScore,
    sampled: MechanismScore | None,
) -> str:
    active = sampled if sampled is not None else noise_free
    mechanism_count = int(active.h1_passed) + int(active.ac_passed)
    if mechanism_count == 2 and active.base_passed and active.rf_passed:
        return "MECHANISM-IDENTIFIABLE-RETINA-SUPPORTED"
    if mechanism_count == 1 and active.rf_passed:
        return "MECHANISM-IDENTIFIABILITY-PARTIAL"
    if not active.rf_passed:
        return "MECHANISM_IDENTIFIABILITY_RF_TRADEOFF"
    return "RF-SUPPORTED-PATHWAYS-NOT-IDENTIFIABLE"


def _by_teacher(
    runs: tuple[MechanismRunEvidence, ...],
) -> dict[str, dict[AblationName, tuple[MechanismRunEvidence, ...]]]:
    names = ("Base", "H1-specific", "AC-specific")
    result = {}
    for name in names:
        rows = tuple(run for run in runs if run.teacher == name)
        result[name] = {
            ablation: tuple(run for run in rows if run.ablation is ablation)
            for ablation in AblationName
        }
    if any(len(rows) != 3 for methods in result.values() for rows in methods.values()):
        raise MechanismScoreError("each teacher and ablation must contain three seeds")
    return result


def _present_h1(full: MechanismRunEvidence, retrained: MechanismRunEvidence) -> bool:
    fraction = (full.no_h1_ce - full.validation_ce) / max(1e-12, full.bias_ce - full.validation_ce)
    return (
        fraction >= 0.20
        and full.gates.h1 >= 0.10
        and full.pathway_cosines["H1"] >= 0.50
        and retrained.validation_ce > full.validation_ce
    )


def _present_ac(full: MechanismRunEvidence, retrained: MechanismRunEvidence) -> bool:
    fraction = (full.no_ac_ce - full.validation_ce) / max(1e-12, full.bias_ce - full.validation_ce)
    return (
        fraction >= 0.20
        and min(full.gates.ac_local, full.gates.ac_transient) >= 0.50
        and full.pathway_cosines["AC"] >= 0.50
        and retrained.validation_ce > full.validation_ce
    )


def _rf_pass(run: MechanismRunEvidence, global_gate: float) -> bool:
    metric = run.rf.metric
    return (
        metric.global_cosine >= global_gate
        and metric.temporal_cosine >= 0.85
        and metric.exact_fraction * 16 >= 12
        and metric.type_polarity_fraction * 16 >= 15
        and run.pathway_sum_error <= 2e-6
        and run.validation_ce < run.bias_ce
    )


__all__ = ["MechanismScore", "final_case", "score_runs"]
