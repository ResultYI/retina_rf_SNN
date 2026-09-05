from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.artifacts import write_csv, write_json
from evaluation.model_comparison.config import ComparisonConfig
from evaluation.model_comparison.presentation import decision_report, write_pareto
from evaluation.model_comparison.types import RunResult


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    root: Path
    config_path: Path
    config: ComparisonConfig
    runs: tuple[RunResult, ...]
    rows: Sequence[Mapping[str, JsonValue]]
    stability: Mapping[str, JsonValue]
    decision: Mapping[str, JsonValue]
    parameters: Mapping[str, JsonValue]
    identity: Mapping[str, JsonValue]


def write_experiment_artifacts(bundle: ArtifactBundle) -> None:
    output = bundle.root / bundle.config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity-manifest.json", bundle.identity)
    output.joinpath("experiment-config.yaml").write_text(
        bundle.config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output / "parameter-counts.json", bundle.parameters)
    write_json(output / "prediction-results.json", _prediction_payload(bundle))
    write_json(output / "rf-results.json", _rf_payload(bundle))
    write_json(output / "stability-results.json", bundle.stability)
    write_csv(
        output / "per-run-metrics.csv",
        _run_fields(),
        tuple(_run_row(run) for run in bundle.runs),
    )
    write_csv(
        output / "per-cell-metrics.csv",
        _cell_fields(),
        tuple(row for run in bundle.runs for row in _cell_rows(run)),
    )
    write_csv(
        output / "model-comparison.csv",
        tuple(bundle.rows[0]),
        bundle.rows,
    )
    write_pareto(output / "pareto.png", bundle.rows)
    output.joinpath("decision-report-zh.md").write_text(
        decision_report(bundle.rows, bundle.decision), encoding="utf-8"
    )


def _prediction_payload(bundle: ArtifactBundle) -> dict[str, JsonValue]:
    return {
        "psth_correlation_status": "PSTH_CORRELATION_NOT_RELIABLE",
        "bits_per_spike_reference": "train-only Bias NLL",
        "models": [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "model",
                    "params",
                    "runs",
                    "val_ce_mean",
                    "val_ce_sd",
                    "sampled_nll_mean",
                    "bits_per_spike_mean",
                    "logit_rmse_mean",
                    "brier_mean",
                }
            }
            for row in bundle.rows
        ],
        "runs": [
            {
                "model": run.model,
                "bank_seed": run.bank_seed,
                "model_seed": run.model_seed,
                "metrics": asdict(run.prediction),
                "training": [asdict(point) for point in run.training],
                "gradients_finite": run.gradients_finite,
            }
            for run in bundle.runs
        ],
    }


def _rf_payload(bundle: ArtifactBundle) -> dict[str, JsonValue]:
    return {
        "estimand": "conditional total-dynamic logit RF",
        "lags": 16,
        "dt_ms": 5.0,
        "models": [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "model",
                    "global_rf_mean",
                    "global_rf_sd",
                    "spatial_mean",
                    "temporal_mean",
                    "rf_norm_error_mean",
                    "center_error_mean",
                    "radius_error_mean",
                    "exact_cell_mean",
                    "nearest_type_polarity_mean",
                    "prototype_centroid_mean",
                }
            }
            for row in bundle.rows
        ],
        "runs": [_rf_run(run) for run in bundle.runs if run.rf is not None],
    }


def _rf_run(run: RunResult) -> dict[str, JsonValue]:
    if run.rf is None:
        raise ValueError("RF payload requires an RF result")
    metric = run.rf.summary.metric
    return {
        "model": run.model,
        "bank_seed": run.bank_seed,
        "model_seed": run.model_seed,
        "global_rf_cosine": metric.global_cosine,
        "signed_spatial_cosine": metric.spatial_cosine,
        "temporal_cosine": metric.temporal_cosine,
        "mean_rf_norm_error": run.rf.summary.mean_norm_error,
        "mean_center_error": metric.geometry.mean_center_error,
        "mean_radius_error": metric.geometry.mean_radius_error,
        "exact_cell_fraction": metric.exact_fraction,
        "nearest_cell_type_polarity_fraction": run.rf.nearest_type_polarity_fraction,
        "type_prototype_centroid_consistency": run.rf.prototype_centroid_fraction,
        "extras": dict(run.extras),
    }


def _run_fields() -> tuple[str, ...]:
    return (
        "model", "bank_seed", "model_seed", "params", "val_ce", "sampled_nll",
        "bits_per_spike", "logit_rmse", "brier", "global_rf", "spatial",
        "temporal", "exact_cell", "nearest_type_polarity", "prototype_centroid",
        "gradients_finite",
    )


def _run_row(run: RunResult) -> dict[str, JsonValue]:
    metric = None if run.rf is None else run.rf.summary.metric
    return {
        "model": run.model,
        "bank_seed": run.bank_seed,
        "model_seed": run.model_seed,
        "params": run.parameter_count,
        "val_ce": run.prediction.teacher_expected_ce,
        "sampled_nll": run.prediction.sampled_nll,
        "bits_per_spike": run.prediction.bits_per_spike,
        "logit_rmse": run.prediction.logit_rmse,
        "brier": run.prediction.brier_score,
        "global_rf": None if metric is None else metric.global_cosine,
        "spatial": None if metric is None else metric.spatial_cosine,
        "temporal": None if metric is None else metric.temporal_cosine,
        "exact_cell": None if metric is None else metric.exact_fraction,
        "nearest_type_polarity": None if run.rf is None else run.rf.nearest_type_polarity_fraction,
        "prototype_centroid": None if run.rf is None else run.rf.prototype_centroid_fraction,
        "gradients_finite": run.gradients_finite,
    }


def _cell_fields() -> tuple[str, ...]:
    return (
        "model", "bank_seed", "model_seed", "cell_id", "expected_ce", "sampled_nll",
        "logit_rmse", "brier", "full_rf_cosine", "spatial_cosine", "temporal_cosine",
        "rf_norm_error", "center_error", "radius_error", "exact_resolved",
        "nearest_type_polarity_resolved", "prototype_centroid_resolved",
    )


def _cell_rows(run: RunResult) -> tuple[dict[str, JsonValue], ...]:
    rows = []
    for index, expected_ce in enumerate(run.prediction.per_cell_expected_ce):
        cell = None if run.rf is None else run.rf.summary.metric.cells[index]
        identity = None if run.rf is None else run.rf.identities[index]
        rows.append(
            {
                "model": run.model,
                "bank_seed": run.bank_seed,
                "model_seed": run.model_seed,
                "cell_id": f"cell-{index}" if cell is None else cell.cell_id,
                "expected_ce": expected_ce,
                "sampled_nll": run.prediction.per_cell_sampled_nll[index],
                "logit_rmse": run.prediction.per_cell_logit_rmse[index],
                "brier": run.prediction.per_cell_brier[index],
                "full_rf_cosine": None if cell is None else cell.full_cosine,
                "spatial_cosine": None if cell is None else cell.spatial_cosine,
                "temporal_cosine": None if cell is None else cell.temporal_cosine,
                "rf_norm_error": None if run.rf is None else run.rf.summary.per_cell_norm_error[index],
                "center_error": None if cell is None else cell.center_error,
                "radius_error": None if cell is None else cell.radius_error,
                "exact_resolved": None if cell is None else cell.exact_resolved,
                "nearest_type_polarity_resolved": None if identity is None else identity.nearest_type_polarity_resolved,
                "prototype_centroid_resolved": None if identity is None else identity.prototype_centroid_resolved,
            }
        )
    return tuple(rows)


__all__ = ["ArtifactBundle", "write_experiment_artifacts"]
