from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from evaluation.mechanistic_retina.mechanism_false_positive import (
    FalsePositiveSummary,
)
from evaluation.mechanistic_retina.mechanism_heldout import HeldoutPathway
from evaluation.mechanistic_retina.mechanism_heldout_metrics import (
    HeldoutSeedMetrics,
    PredictionMetrics,
)
from evaluation.mechanistic_retina.metrics import JsonValue


@dataclass(frozen=True, slots=True)
class SeedSupport:
    seed: int
    full_better_structural: bool
    full_better_clamp: bool
    pathway_rf_supported: bool
    current_supported: bool
    sensitivity_supported: bool
    direction_supported: bool
    present_effect_above_base: bool

    def passed(self) -> bool:
        return all(
            (
                self.full_better_structural,
                self.full_better_clamp,
                self.pathway_rf_supported,
                self.current_supported,
                self.sensitivity_supported,
                self.direction_supported,
                self.present_effect_above_base,
            )
        )


@dataclass(frozen=True, slots=True)
class PathwayDecision:
    pathway: HeldoutPathway
    metrics: tuple[HeldoutSeedMetrics, ...]
    support: tuple[SeedSupport, ...]
    passing_seeds: int
    supported: bool


@dataclass(frozen=True, slots=True)
class FinalDecision:
    h1: PathwayDecision
    ac: PathwayDecision
    false_positive: FalsePositiveSummary
    case: str


def decide_pathway(
    pathway: HeldoutPathway,
    metrics: tuple[HeldoutSeedMetrics, ...],
    false_positive: FalsePositiveSummary,
) -> PathwayDecision:
    floor = torch.finfo(torch.float32).eps
    rows = []
    for value in metrics:
        base = false_positive.for_seed(value.seed)
        base_effect = (
            base.base_h1_clamp_effect
            if pathway is HeldoutPathway.H1
            else base.base_ac_clamp_effect
        )
        rows.append(
            SeedSupport(
                value.seed,
                value.structural_ce_delta > 0.0,
                value.clamp_ce_delta > 0.0,
                value.pathway.rf_cosine >= 0.50,
                value.pathway.current > floor,
                value.pathway.sensitivity > floor,
                value.response_direction_consistent,
                value.clamp_ce_delta > base_effect,
            )
        )
    passing = sum(value.passed() for value in rows)
    return PathwayDecision(pathway, metrics, tuple(rows), passing, passing >= 2)


def final_decision(
    h1: PathwayDecision,
    ac: PathwayDecision,
    false_positive: FalsePositiveSummary,
) -> FinalDecision:
    if h1.supported and ac.supported:
        case = "MECHANISM-IDENTIFIABLE-RETINA-SUPPORTED"
    elif h1.supported or ac.supported:
        case = "MECHANISM-IDENTIFIABILITY-PARTIAL"
    else:
        case = "RF-SUPPORTED-MECHANISM-GENERALIZATION-FAILED"
    return FinalDecision(h1, ac, false_positive, case)


def heldout_payload(decision: PathwayDecision) -> Mapping[str, JsonValue]:
    return {
        "schema": "mechanism-heldout-pathway-v1",
        "pathway": decision.pathway.value,
        "optimizer_steps": 0,
        "criteria": {
            "required_passing_seeds": 2,
            "pathway_rf_cosine_min": 0.50,
            "numerical_floor": torch.finfo(torch.float32).eps,
            "effect_comparison": "present clamp CE delta > paired Base clamp CE delta",
        },
        "passing_seeds": decision.passing_seeds,
        "supported": decision.supported,
        "seeds": [
            _heldout_seed_payload(metric, support)
            for metric, support in zip(
                decision.metrics, decision.support, strict=True
            )
        ],
    }


def write_per_seed_csv(path: Path, decision: FinalDecision) -> None:
    fields = (
        "pathway",
        "seed",
        "full_expected_ce",
        "structural_expected_ce",
        "clamped_expected_ce",
        "structural_ce_delta",
        "clamp_ce_delta",
        "full_logit_rmse",
        "full_response_correlation",
        "activation",
        "current",
        "sensitivity",
        "pathway_rf_norm",
        "pathway_rf_cosine",
        "direction_consistent",
        "base_clamp_effect",
        "seed_passed",
        "optimizer_steps",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pathway in (decision.h1, decision.ac):
            for metric, support in zip(
                pathway.metrics, pathway.support, strict=True
            ):
                base = decision.false_positive.for_seed(metric.seed)
                base_effect = (
                    base.base_h1_clamp_effect
                    if pathway.pathway is HeldoutPathway.H1
                    else base.base_ac_clamp_effect
                )
                writer.writerow(
                    {
                        "pathway": pathway.pathway.value,
                        "seed": metric.seed,
                        "full_expected_ce": metric.full.expected_ce,
                        "structural_expected_ce": metric.structural.expected_ce,
                        "clamped_expected_ce": metric.clamped.expected_ce,
                        "structural_ce_delta": metric.structural_ce_delta,
                        "clamp_ce_delta": metric.clamp_ce_delta,
                        "full_logit_rmse": metric.full.logit_rmse,
                        "full_response_correlation": metric.full.response_correlation,
                        "activation": metric.pathway.activation,
                        "current": metric.pathway.current,
                        "sensitivity": metric.pathway.sensitivity,
                        "pathway_rf_norm": metric.pathway.rf_norm,
                        "pathway_rf_cosine": metric.pathway.rf_cosine,
                        "direction_consistent": metric.response_direction_consistent,
                        "base_clamp_effect": base_effect,
                        "seed_passed": support.passed(),
                        "optimizer_steps": metric.optimizer_steps,
                    }
                )


def _heldout_seed_payload(
    metric: HeldoutSeedMetrics,
    support: SeedSupport,
) -> Mapping[str, JsonValue]:
    return {
        "seed": metric.seed,
        "optimizer_steps": metric.optimizer_steps,
        "prediction": {
            "full": _prediction_payload(metric.full),
            "structural": _prediction_payload(metric.structural),
            "clamped": _prediction_payload(metric.clamped),
            "structural_ce_delta": metric.structural_ce_delta,
            "clamp_ce_delta": metric.clamp_ce_delta,
        },
        "pathway": {
            "activation": metric.pathway.activation,
            "current": metric.pathway.current,
            "sensitivity": metric.pathway.sensitivity,
            "rf_norm": metric.pathway.rf_norm,
            "rf_cosine": metric.pathway.rf_cosine,
            "rf_sha256": metric.pathway.rf_sha256,
            "teacher_component_norm": metric.pathway.teacher_component_norm,
            "teacher_component_sha256": metric.pathway.teacher_component_sha256,
        },
        "diagnostic_responses": [
            {
                "name": response.name,
                "preregistered_name": response.preregistered_name,
                "teacher": response.teacher,
                "full": response.full,
                "structural": response.structural,
                "clamped": response.clamped,
                "direction_cosine": response.direction_cosine,
            }
            for response in metric.responses
        ],
        "response_direction_consistent": metric.response_direction_consistent,
        "support": {
            "full_better_structural": support.full_better_structural,
            "full_better_clamp": support.full_better_clamp,
            "pathway_rf_supported": support.pathway_rf_supported,
            "current_supported": support.current_supported,
            "sensitivity_supported": support.sensitivity_supported,
            "direction_supported": support.direction_supported,
            "present_effect_above_base": support.present_effect_above_base,
            "passed": support.passed(),
        },
    }


def _prediction_payload(value: PredictionMetrics) -> Mapping[str, JsonValue]:
    return {
        "teacher_expected_ce": value.expected_ce,
        "logit_rmse": value.logit_rmse,
        "teacher_response_correlation": value.response_correlation,
    }


__all__ = [
    "FinalDecision",
    "PathwayDecision",
    "SeedSupport",
    "decide_pathway",
    "final_decision",
    "heldout_payload",
    "write_per_seed_csv",
]
