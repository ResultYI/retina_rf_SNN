from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.types import RunResult


MODEL_ORDER = ("Bias", "GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina")


def aggregate_models(
    runs: Sequence[RunResult], stability: Mapping[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    rows = []
    for model in MODEL_ORDER:
        selected = tuple(run for run in runs if run.model == model)
        if not selected:
            continue
        rf_runs = tuple(run for run in selected if run.rf is not None)
        stable = stability[model]
        if not isinstance(stable, dict):
            raise TypeError("stability entry must be a mapping")
        rows.append(
            {
                "model": model,
                "params": selected[0].parameter_count,
                "runs": len(selected),
                "val_ce_mean": _mean(selected, "teacher_expected_ce"),
                "val_ce_sd": _sd(selected, "teacher_expected_ce"),
                "sampled_nll_mean": _mean(selected, "sampled_nll"),
                "bits_per_spike_mean": _mean(selected, "bits_per_spike"),
                "logit_rmse_mean": _mean(selected, "logit_rmse"),
                "brier_mean": _mean(selected, "brier_score"),
                "global_rf_mean": _rf_mean(rf_runs, "global_cosine"),
                "global_rf_sd": _rf_sd(rf_runs, "global_cosine"),
                "spatial_mean": _rf_mean(rf_runs, "spatial_cosine"),
                "temporal_mean": _rf_mean(rf_runs, "temporal_cosine"),
                "rf_norm_error_mean": _norm_mean(rf_runs),
                "center_error_mean": _geometry_mean(rf_runs, "center"),
                "radius_error_mean": _geometry_mean(rf_runs, "radius"),
                "exact_cell_mean": _rf_mean(rf_runs, "exact_fraction"),
                "nearest_type_polarity_mean": _nearest_mean(rf_runs),
                "prototype_centroid_mean": _prototype_mean(rf_runs),
                "stability": stable.get("cross_seed_rf_cosine", stable.get("cross_bank_rf_cosine")),
            }
        )
    return rows


def scientific_decision(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    by_model = {str(row["model"]): row for row in rows}
    main = by_model["Mechanistic Retina"]
    baselines = tuple(by_model[name] for name in ("GLM-SH", "LN-LN", "Graph-TCN"))
    main_ce = float(main["val_ce_mean"])
    main_rf = float(main["global_rf_mean"])
    best_ce = min(float(row["val_ce_mean"]) for row in baselines)
    best_rf = max(float(row["global_rf_mean"]) for row in baselines)
    ce_gap = main_ce - best_ce
    rf_gap = main_rf - best_rf
    prediction_close = ce_gap <= max(0.005, 0.01 * best_ce)
    rf_close = rf_gap >= -0.05
    rf_advantage = rf_gap >= 0.05
    if prediction_close and rf_advantage:
        case = "MECHANISTIC-BALANCED-ADVANTAGE"
        failure = "D"
    elif not prediction_close and rf_advantage:
        case = "MECHANISTIC-RF-INTERPRETABILITY-ADVANTAGE"
        failure = "A"
    elif prediction_close and rf_close:
        case = "MECHANISTIC-PREDICTION-COMPETITIVE"
        failure = "D"
    elif not prediction_close and main_rf > best_rf:
        case = "MECHANISTIC-PARETO-TRADEOFF"
        failure = "E"
    else:
        case = "MECHANISTIC-NO-CLEAR-ADVANTAGE"
        failure = "C" if not prediction_close and not rf_close else "B"
    optimization = _optimization(failure)
    return {
        "case": case,
        "failure_mode": failure,
        "main_ce_gap_to_best_baseline": ce_gap,
        "main_rf_gap_to_best_baseline": rf_gap,
        "clear_architecture_defect": failure in {"A", "B", "C"},
        "authorized_minimal_future_change": optimization,
        "sample_efficiency_authorized": rf_advantage,
        "topology_controls_authorized": case != "MECHANISTIC-NO-CLEAR-ADVANTAGE",
        "natural_image_authorized": case in {
            "MECHANISTIC-BALANCED-ADVANTAGE",
            "MECHANISTIC-PREDICTION-COMPETITIVE",
            "MECHANISTIC-RF-INTERPRETABILITY-ADVANTAGE",
            "MECHANISTIC-PARETO-TRADEOFF",
        },
    }


def _optimization(failure: str) -> JsonValue:
    if failure == "A":
        return "pathway-local bounded grouped/depthwise TCN operator, neutral initialization, no direct logit"
    if failure == "B":
        return "diagnose the named pathway, then change only its basis/support/sharing/state parameterization"
    if failure == "C":
        return "one separately specified core revision after a falsifiable failure-source diagnosis"
    return None


def _mean(runs: Sequence[RunResult], field: str) -> float:
    return statistics.fmean(float(getattr(run.prediction, field)) for run in runs)


def _sd(runs: Sequence[RunResult], field: str) -> float:
    values = tuple(float(getattr(run.prediction, field)) for run in runs)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _rf_mean(runs: Sequence[RunResult], field: str) -> float | None:
    values = tuple(float(getattr(run.rf.summary.metric, field)) for run in runs if run.rf is not None)
    return statistics.fmean(values) if values else None


def _rf_sd(runs: Sequence[RunResult], field: str) -> float | None:
    values = tuple(float(getattr(run.rf.summary.metric, field)) for run in runs if run.rf is not None)
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _norm_mean(runs: Sequence[RunResult]) -> float | None:
    values = tuple(run.rf.summary.mean_norm_error for run in runs if run.rf is not None)
    return statistics.fmean(values) if values else None


def _geometry_mean(runs: Sequence[RunResult], kind: str) -> float | None:
    values = tuple(
        run.rf.summary.metric.geometry.mean_center_error
        if kind == "center"
        else run.rf.summary.metric.geometry.mean_radius_error
        for run in runs
        if run.rf is not None
    )
    return statistics.fmean(values) if values else None


def _nearest_mean(runs: Sequence[RunResult]) -> float | None:
    values = tuple(run.rf.nearest_type_polarity_fraction for run in runs if run.rf is not None)
    return statistics.fmean(values) if values else None


def _prototype_mean(runs: Sequence[RunResult]) -> float | None:
    values = tuple(run.rf.prototype_centroid_fraction for run in runs if run.rf is not None)
    return statistics.fmean(values) if values else None


__all__ = ["MODEL_ORDER", "aggregate_models", "scientific_decision"]
