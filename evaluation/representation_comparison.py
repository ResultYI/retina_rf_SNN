from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.representation_diagnostics import RepresentationDiagnostics


@dataclass(frozen=True, slots=True)
class SourceRepresentationDelta:
    source_id: str
    fixed_mse_delta: float
    probe_mse_delta: float
    generator_probe_mse_delta: float = float("nan")


@dataclass(frozen=True, slots=True)
class RepresentationComparison:
    current_mse_delta: float
    fixed_mse_delta: float
    probe_mse_delta: float
    probe_source_cv_mse_delta: float
    improved_fixed_source_count: int
    total_source_count: int
    sources: tuple[SourceRepresentationDelta, ...]
    generator_probe_mse_delta: float = float("nan")
    generator_probe_source_cv_mse_delta: float = float("nan")
    improved_generator_source_count: int = 0


def compare_representation_diagnostics(
    initial: RepresentationDiagnostics,
    selected: RepresentationDiagnostics,
) -> RepresentationComparison:
    selected_sources = {
        metric.source_id: metric for metric in selected.source_metrics
    }
    deltas = tuple(
        SourceRepresentationDelta(
            source_id=metric.source_id,
            fixed_mse_delta=(
                selected_sources[metric.source_id].fixed_calibrated_decoder_mse
                - metric.fixed_calibrated_decoder_mse
            ),
            probe_mse_delta=(
                selected_sources[metric.source_id].posthoc_tied_decoder_probe_mse
                - metric.posthoc_tied_decoder_probe_mse
            ),
            generator_probe_mse_delta=(
                selected_sources[metric.source_id].posthoc_generator_probe_mse
                - metric.posthoc_generator_probe_mse
            ),
        )
        for metric in initial.source_metrics
    )
    return RepresentationComparison(
        current_mse_delta=(
            selected.current_decoder.mse - initial.current_decoder.mse
        ),
        fixed_mse_delta=(
            selected.fixed_calibrated_decoder.mse
            - initial.fixed_calibrated_decoder.mse
        ),
        probe_mse_delta=(
            selected.posthoc_tied_decoder_probe.mse
            - initial.posthoc_tied_decoder_probe.mse
        ),
        probe_source_cv_mse_delta=(
            selected.posthoc_tied_decoder_probe_source_cv_mse
            - initial.posthoc_tied_decoder_probe_source_cv_mse
        ),
        improved_fixed_source_count=sum(
            delta.fixed_mse_delta < 0.0 for delta in deltas
        ),
        total_source_count=len(deltas),
        sources=deltas,
        generator_probe_mse_delta=_generator_metric_delta(initial, selected),
        generator_probe_source_cv_mse_delta=(
            selected.posthoc_generator_probe_source_cv_mse
            - initial.posthoc_generator_probe_source_cv_mse
        ),
        improved_generator_source_count=sum(
            delta.generator_probe_mse_delta < 0.0 for delta in deltas
        ),
    )


def _generator_metric_delta(
    initial: RepresentationDiagnostics,
    selected: RepresentationDiagnostics,
) -> float:
    return selected.posthoc_generator_probe.mse - initial.posthoc_generator_probe.mse


__all__ = [
    "RepresentationComparison",
    "SourceRepresentationDelta",
    "compare_representation_diagnostics",
]
