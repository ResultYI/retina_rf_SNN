from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import statistics

from evaluation.mechanistic_retina.mechanism_replay_types import ReplayRunSet
from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    MechanismRunEvidence,
)
from evaluation.mechanistic_retina.metrics import JsonValue


@dataclass(frozen=True, slots=True)
class FalsePositiveSeed:
    seed: int
    base_h1_gate: float
    base_ac_gate: float
    base_h1_rf_norm: float
    base_ac_rf_norm: float
    base_h1_clamp_effect: float
    base_ac_clamp_effect: float
    h1_false_ac_gate_ratio: float
    h1_false_ac_rf_ratio: float
    h1_false_ac_clamp_ratio: float
    ac_false_h1_gate_ratio: float
    ac_false_h1_rf_ratio: float
    ac_false_h1_clamp_ratio: float


@dataclass(frozen=True, slots=True)
class FalsePositiveSummary:
    seeds: tuple[FalsePositiveSeed, ...]

    def for_seed(self, seed: int) -> FalsePositiveSeed:
        for value in self.seeds:
            if value.seed == seed:
                return value
        raise KeyError(seed)


def evaluate_false_positives(run_set: ReplayRunSet) -> FalsePositiveSummary:
    rows = []
    for seed in (19, 20, 21):
        base = _full(run_set, "Base", seed)
        h1 = _full(run_set, "H1-specific", seed)
        ac = _full(run_set, "AC-specific", seed)
        base_h1_clamp = abs(base.no_h1_ce - base.validation_ce)
        base_ac_clamp = abs(base.no_ac_ce - base.validation_ce)
        h1_present = abs(h1.no_h1_ce - h1.validation_ce)
        ac_present = abs(ac.no_ac_ce - ac.validation_ce)
        rows.append(
            FalsePositiveSeed(
                seed,
                base.gates.h1,
                max(base.gates.ac_local, base.gates.ac_transient),
                base.pathway_norms["H1"],
                base.pathway_norms["AC"],
                base_h1_clamp,
                base_ac_clamp,
                max(h1.gates.ac_local, h1.gates.ac_transient)
                / max(1e-12, h1.gates.h1),
                h1.pathway_norms["AC"] / max(1e-12, h1.pathway_norms["H1"]),
                abs(h1.no_ac_ce - h1.validation_ce) / max(1e-12, h1_present),
                ac.gates.h1
                / max(1e-12, max(ac.gates.ac_local, ac.gates.ac_transient)),
                ac.pathway_norms["H1"] / max(1e-12, ac.pathway_norms["AC"]),
                abs(ac.no_h1_ce - ac.validation_ce) / max(1e-12, ac_present),
            )
        )
    return FalsePositiveSummary(tuple(rows))


def false_positive_payload(
    summary: FalsePositiveSummary,
) -> Mapping[str, JsonValue]:
    medians = {
        "h1_teacher_false_ac": {
            "gate_ratio": _median(summary, "h1_false_ac_gate_ratio"),
            "rf_ratio": _median(summary, "h1_false_ac_rf_ratio"),
            "clamp_ratio": _median(summary, "h1_false_ac_clamp_ratio"),
        },
        "ac_teacher_false_h1": {
            "gate_ratio": _median(summary, "ac_false_h1_gate_ratio"),
            "rf_ratio": _median(summary, "ac_false_h1_rf_ratio"),
            "clamp_ratio": _median(summary, "ac_false_h1_clamp_ratio"),
        },
    }
    return {
        "schema": "mechanism-heldout-false-positive-v1",
        "prior_reference": {
            "h1_teacher_false_ac": {
                "gate_ratio": 0.0243,
                "rf_ratio": 0.0163,
                "clamp_ratio": 5e-5,
            },
            "ac_teacher_false_h1": {
                "gate_ratio": 0.1538,
                "rf_ratio": 0.0044,
                "clamp_ratio": 4.5e-4,
            },
        },
        "median": medians,
        "direction_maintained": all(
            value < 1.0
            for group in medians.values()
            for value in group.values()
        ),
        "seeds": [_seed_payload(value) for value in summary.seeds],
    }


def _full(run_set: ReplayRunSet, teacher: str, seed: int) -> MechanismRunEvidence:
    for run in run_set.runs:
        if run.teacher == teacher and run.ablation is AblationName.FULL and run.seed == seed:
            return run
    raise KeyError((teacher, seed))


def _median(summary: FalsePositiveSummary, field: str) -> float:
    return float(statistics.median(getattr(value, field) for value in summary.seeds))


def _seed_payload(value: FalsePositiveSeed) -> Mapping[str, JsonValue]:
    return {
        "seed": value.seed,
        "base": {
            "h1_gate": value.base_h1_gate,
            "ac_gate": value.base_ac_gate,
            "h1_rf_norm": value.base_h1_rf_norm,
            "ac_rf_norm": value.base_ac_rf_norm,
            "h1_clamp_effect": value.base_h1_clamp_effect,
            "ac_clamp_effect": value.base_ac_clamp_effect,
        },
        "h1_teacher_false_ac": {
            "gate_ratio": value.h1_false_ac_gate_ratio,
            "rf_ratio": value.h1_false_ac_rf_ratio,
            "clamp_ratio": value.h1_false_ac_clamp_ratio,
        },
        "ac_teacher_false_h1": {
            "gate_ratio": value.ac_false_h1_gate_ratio,
            "rf_ratio": value.ac_false_h1_rf_ratio,
            "clamp_ratio": value.ac_false_h1_clamp_ratio,
        },
    }


__all__ = [
    "FalsePositiveSeed",
    "FalsePositiveSummary",
    "evaluate_false_positives",
    "false_positive_payload",
]
