from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Final

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.sample_efficiency_reporting import (
    AggregateMetricRow,
    MetricRow,
    Profile,
    aggregate_metric_rows,
)


FIGURE_NAMES: Final = frozenset({"sample-efficiency-rf.png", "sample-efficiency-ce.png"})
COMPARISON_PROFILES: Final = (Profile.ARCHITECTURE_SIZE, Profile.ACTIVE_DOF)


@dataclass(frozen=True, slots=True)
class PresentationError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class _FigureSpec:
    path: Path
    points: Sequence[MetricRow]
    aggregates: Sequence[AggregateMetricRow]
    field: str
    ylabel: str


def write_sample_efficiency_figures(output_dir: Path, rows: Sequence[MetricRow]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and path.name not in FIGURE_NAMES
    }
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise PresentationError("UNEXPECTED_SAMPLE_EFFICIENCY_FIGURE", names)
    aggregates = aggregate_metric_rows(rows)
    _write_metric_figure(_FigureSpec(
        output_dir / "sample-efficiency-rf.png",
        tuple(row for row in rows if row.global_rf is not None),
        aggregates,
        "global_rf",
        "Global RF cosine",
    ))
    _write_metric_figure(_FigureSpec(
        output_dir / "sample-efficiency-ce.png",
        rows,
        aggregates,
        "val_ce",
        "Validation CE",
    ))


def build_decision_payload(rows: Sequence[AggregateMetricRow]) -> Mapping[str, JsonValue]:
    regimes: list[JsonValue] = []
    for profile in COMPARISON_PROFILES:
        selected = tuple(
            row
            for row in rows
            if row.profile is profile or row.profile is Profile.SHARED_REFERENCE
        )
        fractions = sorted({row.fraction for row in selected})
        comparisons: list[JsonValue] = []
        for fraction in fractions:
            comparison = _comparison_for_fraction(profile, fraction, selected)
            if comparison:
                comparisons.append(comparison)
        regimes.append(
            {
                "profile": profile.value,
                "comparisons": comparisons,
                "outcome": _profile_outcome(comparisons),
            }
        )
    return {
        "claim_policy": "negative and mixed outcomes are allowed",
        "regimes": regimes,
    }


def sample_efficiency_report(
    rows: Sequence[AggregateMetricRow], decision: Mapping[str, JsonValue]
) -> str:
    lines = [
        "# Candidate0 T=2 sample-efficiency report",
        "",
        "Decision policy: negative and mixed outcomes are allowed.",
        "",
        "| Fraction | Profile | Model | CE | Bits/spike | RF | Exact cell | Cross-seed | Cross-bank |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.fraction:.2f} | {row.profile.value} | {row.model} | "
            f"{row.val_ce_mean:.6f} | {row.bits_per_spike_mean:.4f} | "
            f"{_fmt(row.global_rf_mean)} | {_fmt(row.exact_cell_mean)} | "
            f"{_fmt(row.cross_seed_rf_mean)} | {_fmt(row.cross_bank_rf_mean)} |"
        )
    lines.extend(("", "Regime outcomes:"))
    regimes = decision["regimes"]
    if isinstance(regimes, list):
        for regime in regimes:
            if isinstance(regime, dict):
                lines.append(f"- {regime['profile']}: {regime['outcome']}")
    return "\n".join(lines) + "\n"


def _comparison_for_fraction(
    profile: Profile, fraction: float, rows: Sequence[AggregateMetricRow]
) -> dict[str, JsonValue] | None:
    mechanistic_matches = tuple(
        row
        for row in rows
        if row.fraction == fraction
        and row.model == "Mechanistic Retina"
        and row.profile is profile
    )
    mechanistic = mechanistic_matches[0] if mechanistic_matches else None
    if mechanistic is None or mechanistic.global_rf_mean is None:
        return None
    baselines = tuple(
        row
        for row in rows
        if row.fraction == fraction
        and row.model != "Bias"
        and row.model != "Mechanistic Retina"
        and (row.profile is profile or row.profile is Profile.SHARED_REFERENCE)
        and row.global_rf_mean is not None
    )
    if not baselines:
        return None
    best_ce = min(baselines, key=lambda row: row.val_ce_mean)
    best_rf = max(baselines, key=lambda row: row.global_rf_mean or float("-inf"))
    ce_gap = mechanistic.val_ce_mean - best_ce.val_ce_mean
    rf_gap = mechanistic.global_rf_mean - (best_rf.global_rf_mean or 0.0)
    ce_better = ce_gap <= 0.0
    rf_better = rf_gap >= 0.0
    return {
        "fraction": fraction,
        "mechanistic_ce_gap_to_best": ce_gap,
        "mechanistic_rf_gap_to_best": rf_gap,
        "best_ce_baseline": best_ce.model,
        "best_rf_baseline": best_rf.model,
        "outcome": _comparison_outcome(ce_better, rf_better),
    }


def _comparison_outcome(ce_better: bool, rf_better: bool) -> str:
    if ce_better and rf_better:
        return "positive"
    if ce_better or rf_better:
        return "mixed"
    return "negative"


def _profile_outcome(comparisons: Sequence[JsonValue]) -> str:
    outcomes = {
        str(comparison["outcome"])
        for comparison in comparisons
        if isinstance(comparison, dict) and "outcome" in comparison
    }
    if "positive" in outcomes and len(outcomes) == 1:
        return "positive"
    if outcomes:
        return "mixed" if "mixed" in outcomes or len(outcomes) > 1 else "negative"
    return "no-comparison"


def _write_metric_figure(spec: _FigureSpec) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for model, profile in _series(spec.points):
        selected_points = tuple(
            row for row in spec.points if row.model == model and row.profile is profile
        )
        selected_aggregates = tuple(
            row for row in spec.aggregates if row.model == model and row.profile is profile
        )
        label = model if profile is Profile.SHARED_REFERENCE else f"{model} ({profile.value})"
        axis.scatter(
            [100.0 * row.fraction for row in selected_points],
            [_metric(row, spec.field) for row in selected_points],
            alpha=0.35,
            s=18,
        )
        axis.errorbar(
            [100.0 * row.fraction for row in selected_aggregates],
            [_aggregate_metric(row, spec.field) for row in selected_aggregates],
            yerr=[_aggregate_variation(row, spec.field) for row in selected_aggregates],
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=label,
        )
    axis.set_xlabel("Training fraction (%)")
    axis.set_ylabel(spec.ylabel)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(spec.path, dpi=180)
    plt.close(figure)


def _series(rows: Sequence[MetricRow]) -> tuple[tuple[str, Profile], ...]:
    pairs = {(row.model, row.profile) for row in rows}
    return tuple(sorted(pairs, key=lambda pair: (pair[0], pair[1].value)))


def _metric(row: MetricRow, field: str) -> float:
    value = row.global_rf if field == "global_rf" else row.val_ce
    if value is None:
        raise PresentationError("MISSING_PLOT_VALUE", row.model)
    return value


def _aggregate_metric(row: AggregateMetricRow, field: str) -> float:
    value = row.global_rf_mean if field == "global_rf" else row.val_ce_mean
    if value is None:
        raise PresentationError("MISSING_PLOT_VALUE", row.model)
    return value


def _aggregate_variation(row: AggregateMetricRow, field: str) -> float:
    value = row.global_rf_sd if field == "global_rf" else row.val_ce_sd
    return 0.0 if value is None else value


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


__all__ = [
    "PresentationError",
    "build_decision_payload",
    "sample_efficiency_report",
    "write_sample_efficiency_figures",
]
