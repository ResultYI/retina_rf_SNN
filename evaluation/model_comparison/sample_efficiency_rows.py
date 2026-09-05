from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.sample_efficiency_reporting import (
    AggregateMetricRow,
    MetricRow,
    Profile,
    aggregate_metric_rows,
)
from evaluation.model_comparison.sample_efficiency_reuse import (
    CanonicalReuseRequest,
    CanonicalReuseRow,
    load_canonical_reuse_rows,
)
from evaluation.model_comparison.stability import stability_payload
from evaluation.model_comparison.types import RunResult


@dataclass(frozen=True, slots=True)
class ProfiledRunResult:
    run: RunResult
    profile: Profile
    reuse_status: str
    source_run_id: str


def profiled_run(run: RunResult, profile: Profile) -> ProfiledRunResult:
    return ProfiledRunResult(
        run,
        profile,
        "trained",
        f"sample-efficiency:{run.model}:bank-{run.bank_seed}:seed-{run.model_seed or 'none'}",
    )


def metric_rows_from_profiled(
    fraction: float, train_count: int, profiled: Sequence[ProfiledRunResult]
) -> tuple[MetricRow, ...]:
    stability = _stability(profiled)
    return tuple(
        _metric_row(item, fraction, train_count, stability) for item in profiled
    )


def canonical_metric_rows(root: Path) -> tuple[MetricRow, ...]:
    rows = load_canonical_reuse_rows(CanonicalReuseRequest(root=root))
    path = root / ".omo/evidence/canonical-model-comparison-t2/stability-results.json"
    stability = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_canonical_row(row, stability) for row in rows)


def comparison_rows(rows: Sequence[MetricRow]) -> tuple[MetricRow, ...]:
    active_view = tuple(
        replace(
            row,
            profile=Profile.ACTIVE_DOF,
            source_run_id=row.source_run_id + ":active-dof-view",
        )
        for row in rows
        if row.model == "Mechanistic Retina"
    )
    return tuple(rows) + active_view


def row_fields() -> tuple[str, ...]:
    return tuple(row_payload(fixture_row_provider(0.25, 28)[0]))


def aggregate_fields() -> tuple[str, ...]:
    sample = aggregate_metric_rows(comparison_rows(fixture_row_provider(0.25, 28)))[0]
    return tuple(aggregate_payload(sample))


def row_payload(row: MetricRow) -> Mapping[str, JsonValue]:
    return {
        "fraction": row.fraction,
        "train_stimuli": row.train_stimuli,
        "model": row.model,
        "profile": row.profile.value,
        "bank_seed": row.bank_seed,
        "model_seed": row.model_seed,
        "parameter_count": row.parameter_count,
        "val_ce": row.val_ce,
        "sampled_nll": row.sampled_nll,
        "bits_per_spike": row.bits_per_spike,
        "global_rf": row.global_rf,
        "spatial_rf": row.spatial_rf,
        "temporal_rf": row.temporal_rf,
        "exact_cell": row.exact_cell,
        "nearest_type_polarity": row.nearest_type_polarity,
        "prototype_centroid": row.prototype_centroid,
        "cross_seed_rf": row.cross_seed_rf,
        "cross_bank_rf": row.cross_bank_rf,
        "reuse_status": row.reuse_status,
        "source_run_id": row.source_run_id,
    }


def aggregate_payload(row: AggregateMetricRow) -> Mapping[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for field in fields(row):
        value = getattr(row, field.name)
        payload[field.name] = value.value if isinstance(value, Profile) else value
    return payload


def parse_row(raw: Mapping[str, JsonValue]) -> MetricRow:
    return MetricRow(
        float(raw["fraction"]),
        int(raw["train_stimuli"]),
        str(raw["model"]),
        Profile(str(raw["profile"])),
        int(raw["bank_seed"]),
        None if raw["model_seed"] is None else int(raw["model_seed"]),
        int(raw["parameter_count"]),
        float(raw["val_ce"]),
        float(raw["sampled_nll"]),
        float(raw["bits_per_spike"]),
        _maybe_float(raw["global_rf"]),
        _maybe_float(raw["spatial_rf"]),
        _maybe_float(raw["temporal_rf"]),
        _maybe_float(raw["exact_cell"]),
        _maybe_float(raw["nearest_type_polarity"]),
        _maybe_float(raw["prototype_centroid"]),
        _maybe_float(raw["cross_seed_rf"]),
        _maybe_float(raw["cross_bank_rf"]),
        str(raw["reuse_status"]),
        str(raw["source_run_id"]),
    )


def fixture_row_provider(fraction: float, train_count: int) -> tuple[MetricRow, ...]:
    rows: list[MetricRow] = []
    rows.extend(
        _fixture_bank_rows(
            fraction, train_count, "Bias", Profile.SHARED_REFERENCE, None, 16, None
        )
    )
    rows.extend(
        _fixture_bank_rows(
            fraction, train_count, "GLM-SH", Profile.SHARED_REFERENCE, None, 7504, 0.12
        )
    )
    for profile, ln_params, graph_params in (
        (Profile.ARCHITECTURE_SIZE, 240, 270),
        (Profile.ACTIVE_DOF, 160, 144),
    ):
        rows.extend(
            _fixture_seed_rows(fraction, train_count, "LN-LN", profile, ln_params, 0.72)
        )
        rows.extend(
            _fixture_seed_rows(
                fraction, train_count, "Graph-TCN", profile, graph_params, 0.42
            )
        )
    rows.extend(
        _fixture_seed_rows(
            fraction,
            train_count,
            "Mechanistic Retina",
            Profile.ARCHITECTURE_SIZE,
            264,
            0.88,
        )
    )
    if fraction == 1.0:
        reused = tuple(row for row in rows if row.profile is not Profile.ACTIVE_DOF)
        trained = tuple(row for row in rows if row.profile is Profile.ACTIVE_DOF)
        return tuple(replace(row, reuse_status="reused") for row in reused) + trained
    return tuple(rows)


def _stability(
    profiled: Sequence[ProfiledRunResult],
) -> Mapping[tuple[str, Profile], Mapping[str, JsonValue]]:
    payload: dict[tuple[str, Profile], Mapping[str, JsonValue]] = {}
    for key in sorted({(item.run.model, item.profile) for item in profiled}):
        selected = tuple(
            item.run for item in profiled if (item.run.model, item.profile) == key
        )
        payload[key] = stability_payload(selected)[key[0]]
    return payload


def _metric_row(
    item: ProfiledRunResult,
    fraction: float,
    train_count: int,
    stability: Mapping[tuple[str, Profile], Mapping[str, JsonValue]],
) -> MetricRow:
    run = item.run
    metric = None if run.rf is None else run.rf.summary.metric
    entry = stability.get((run.model, item.profile), {})
    return MetricRow(
        fraction,
        train_count,
        run.model,
        item.profile,
        run.bank_seed,
        run.model_seed,
        run.parameter_count,
        run.prediction.teacher_expected_ce,
        run.prediction.sampled_nll,
        run.prediction.bits_per_spike,
        None if metric is None else metric.global_cosine,
        None if metric is None else metric.spatial_cosine,
        None if metric is None else metric.temporal_cosine,
        None if metric is None else metric.exact_fraction,
        None if run.rf is None else run.rf.nearest_type_polarity_fraction,
        None if run.rf is None else run.rf.prototype_centroid_fraction,
        _optional_float(entry.get("cross_seed_rf_cosine")),
        _optional_float(entry.get("cross_bank_rf_cosine")),
        item.reuse_status,
        item.source_run_id,
    )


def _canonical_row(
    row: CanonicalReuseRow, stability: Mapping[str, JsonValue]
) -> MetricRow:
    entry = stability.get(row.model, {})
    return MetricRow(
        row.fraction,
        row.count,
        row.model,
        row.fairness_regime,
        row.bank_seed,
        row.model_seed,
        row.parameter_count,
        row.val_ce,
        row.sampled_nll,
        row.bits_per_spike,
        row.global_rf,
        row.spatial,
        row.temporal,
        row.exact_cell,
        row.nearest_type_polarity,
        row.prototype_centroid,
        (
            _optional_float(entry.get("cross_seed_rf_cosine"))
            if isinstance(entry, dict)
            else None
        ),
        (
            _optional_float(entry.get("cross_bank_rf_cosine"))
            if isinstance(entry, dict)
            else None
        ),
        "reused",
        row.source_run_id,
    )


def _optional_float(value: JsonValue | None) -> float | None:
    return value if isinstance(value, float) else None


def _maybe_float(value: JsonValue) -> float | None:
    return None if value is None or value == "" else float(value)


def _fixture_bank_rows(
    fraction: float,
    train_count: int,
    model: str,
    profile: Profile,
    model_seed: int | None,
    params: int,
    rf: float | None,
) -> tuple[MetricRow, ...]:
    return tuple(
        _fixture_row(
            fraction, train_count, model, profile, seed, model_seed, params, rf
        )
        for seed in (31001, 31002, 31003)
    )


def _fixture_seed_rows(
    fraction: float,
    train_count: int,
    model: str,
    profile: Profile,
    params: int,
    rf: float,
) -> tuple[MetricRow, ...]:
    return tuple(
        _fixture_row(fraction, train_count, model, profile, bank, seed, params, rf)
        for bank in (31001, 31002, 31003)
        for seed in (19, 20, 21)
    )


def _fixture_row(
    fraction: float,
    train_count: int,
    model: str,
    profile: Profile,
    bank_seed: int,
    model_seed: int | None,
    params: int,
    rf: float | None,
) -> MetricRow:
    ce = 0.5 - (0.1 * fraction) - (0.0 if rf is None else rf / 100.0)
    return MetricRow(
        fraction,
        train_count,
        model,
        profile,
        bank_seed,
        model_seed,
        params,
        ce,
        ce + 0.01,
        0.1 + fraction,
        rf,
        None if rf is None else rf - 0.01,
        None if rf is None else rf - 0.02,
        None if rf is None else 0.5,
        None if rf is None else 0.8,
        None if rf is None else 0.7,
        None if rf is None else rf - 0.03,
        None if rf is None else rf - 0.04,
        "trained",
        f"fixture:{model}:{profile.value}:{bank_seed}:{model_seed}",
    )


__all__ = [
    "ProfiledRunResult",
    "aggregate_fields",
    "aggregate_payload",
    "canonical_metric_rows",
    "comparison_rows",
    "fixture_row_provider",
    "metric_rows_from_profiled",
    "parse_row",
    "profiled_run",
    "row_fields",
    "row_payload",
]
