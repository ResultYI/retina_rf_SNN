from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from evaluation.global_probe import (
    GlobalProbePair,
    GlobalReadoutGeometry,
    GlobalSourceMSE,
    fit_global_probe_pair,
)
from evaluation.representation_diagnostics import (
    DecoderExamples,
)
from training.state import (
    RepresentationSelectionMetrics,
)


@dataclass(frozen=True, slots=True)
class RepresentationSelectionInputs:
    train_examples: DecoderExamples
    validation_examples: DecoderExamples
    geometry: GlobalReadoutGeometry
    fixed_validation_mse: float


@dataclass(frozen=True, slots=True)
class RepresentationSelectionSnapshot:
    probes: GlobalProbePair
    metrics: RepresentationSelectionMetrics
    fixed_validation_mse: float
    improved_rate_sources: int
    improved_generator_sources: int


@dataclass(frozen=True, slots=True)
class RepresentationSelectionLog:
    global_rate_source_cv_mse: float
    global_generator_source_cv_mse: float
    global_rate_validation_mse: float
    global_generator_validation_mse: float
    global_rate_source_cv_ratio: float
    global_generator_source_cv_ratio: float
    fixed_validation_ratio: float
    improved_rate_sources: int
    improved_generator_sources: int
    best_representation_event: bool


@torch.no_grad()
def evaluate_representation_selection(
    inputs: RepresentationSelectionInputs,
    baseline: RepresentationSelectionSnapshot | None,
) -> RepresentationSelectionSnapshot:
    probes = fit_global_probe_pair(
        inputs.train_examples,
        inputs.validation_examples,
        inputs.geometry,
    )
    if baseline is None:
        metrics = RepresentationSelectionMetrics(1.0, 1.0, 1.0)
        improved_rate_sources = 0
        improved_generator_sources = 0
    else:
        metrics = RepresentationSelectionMetrics(
            rate_source_cv_ratio=(
                probes.rate.source_cv_mse
                / max(baseline.probes.rate.source_cv_mse, 1e-12)
            ),
            generator_source_cv_ratio=(
                probes.generator.source_cv_mse
                / max(baseline.probes.generator.source_cv_mse, 1e-12)
            ),
            fixed_validation_ratio=(
                inputs.fixed_validation_mse
                / max(baseline.fixed_validation_mse, 1e-12)
            ),
        )
        improved_rate_sources = _improved_source_count(
            probes.rate.validation_source_mse,
            baseline.probes.rate.validation_source_mse,
        )
        improved_generator_sources = _improved_source_count(
            probes.generator.validation_source_mse,
            baseline.probes.generator.validation_source_mse,
        )
    return RepresentationSelectionSnapshot(
        probes=probes,
        metrics=metrics,
        fixed_validation_mse=inputs.fixed_validation_mse,
        improved_rate_sources=improved_rate_sources,
        improved_generator_sources=improved_generator_sources,
    )


def selection_log(
    snapshot: RepresentationSelectionSnapshot,
    best_representation_event: bool,
) -> RepresentationSelectionLog:
    return RepresentationSelectionLog(
        global_rate_source_cv_mse=snapshot.probes.rate.source_cv_mse,
        global_generator_source_cv_mse=(
            snapshot.probes.generator.source_cv_mse
        ),
        global_rate_validation_mse=snapshot.probes.rate.validation_mse,
        global_generator_validation_mse=(
            snapshot.probes.generator.validation_mse
        ),
        global_rate_source_cv_ratio=(
            snapshot.metrics.rate_source_cv_ratio
        ),
        global_generator_source_cv_ratio=(
            snapshot.metrics.generator_source_cv_ratio
        ),
        fixed_validation_ratio=snapshot.metrics.fixed_validation_ratio,
        improved_rate_sources=snapshot.improved_rate_sources,
        improved_generator_sources=snapshot.improved_generator_sources,
        best_representation_event=best_representation_event,
    )


def _improved_source_count(
    current: Sequence[GlobalSourceMSE],
    baseline: Sequence[GlobalSourceMSE],
) -> int:
    baseline_by_source = {row.source_id: row.mse for row in baseline}
    return sum(
        row.mse < baseline_by_source[row.source_id]
        for row in current
    )


__all__ = [
    "RepresentationSelectionInputs",
    "RepresentationSelectionLog",
    "RepresentationSelectionSnapshot",
    "evaluate_representation_selection",
    "selection_log",
]
