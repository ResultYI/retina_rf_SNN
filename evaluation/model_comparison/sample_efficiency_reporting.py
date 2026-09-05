from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import math
import statistics
from collections.abc import Sequence
from typing import Final


MODEL_ORDER: Final = ("Bias", "GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina")
RF_MODELS: Final = frozenset({"GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina"})
PROFILED_MODELS: Final = frozenset({"LN-LN", "Graph-TCN", "Mechanistic Retina"})
SHARED_MODELS: Final = frozenset({"Bias", "GLM-SH"})
SEEDED_RF_MODELS: Final = frozenset({"LN-LN", "Graph-TCN", "Mechanistic Retina"})


@unique
class Profile(StrEnum):
    ARCHITECTURE_SIZE = "architecture-size"
    ACTIVE_DOF = "optimizer-listed-count"
    SHARED_REFERENCE = "shared-reference"


@dataclass(frozen=True, slots=True)
class ReportingSchemaError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class MetricRow:
    fraction: float
    train_stimuli: int
    model: str
    profile: Profile
    bank_seed: int
    model_seed: int | None
    parameter_count: int
    val_ce: float
    sampled_nll: float
    bits_per_spike: float
    global_rf: float | None
    spatial_rf: float | None
    temporal_rf: float | None
    exact_cell: float | None
    nearest_type_polarity: float | None
    prototype_centroid: float | None
    cross_seed_rf: float | None
    cross_bank_rf: float | None
    reuse_status: str
    source_run_id: str


@dataclass(frozen=True, slots=True)
class AggregateMetricRow:
    fraction: float
    train_stimuli: int
    model: str
    profile: Profile
    parameter_count: int
    runs: int
    val_ce_mean: float
    val_ce_sd: float
    sampled_nll_mean: float
    bits_per_spike_mean: float
    global_rf_mean: float | None
    global_rf_sd: float | None
    spatial_rf_mean: float | None
    temporal_rf_mean: float | None
    exact_cell_mean: float | None
    nearest_type_polarity_mean: float | None
    prototype_centroid_mean: float | None
    cross_seed_rf_mean: float | None
    cross_bank_rf_mean: float | None
    reused_runs: int
    trained_runs: int


@dataclass(frozen=True, slots=True)
class _MetricIdentity:
    fraction: float
    model: str
    profile: Profile
    bank_seed: int
    model_seed: int | None


@dataclass(frozen=True, slots=True)
class _AggregateKey:
    fraction: float
    model: str
    profile: Profile


def aggregate_metric_rows(rows: Sequence[MetricRow]) -> tuple[AggregateMetricRow, ...]:
    _validate_rows(rows)
    grouped: dict[_AggregateKey, list[MetricRow]] = {}
    for row in rows:
        key = _AggregateKey(row.fraction, row.model, row.profile)
        grouped.setdefault(key, []).append(row)
    return tuple(_aggregate(grouped[key]) for key in sorted(grouped, key=_sort_key))


def profile_label(profile: Profile) -> str:
    return profile.value


def _validate_rows(rows: Sequence[MetricRow]) -> None:
    identities: set[_MetricIdentity] = set()
    for row in rows:
        _validate_row(row)
        identity = _MetricIdentity(
            row.fraction, row.model, row.profile, row.bank_seed, row.model_seed
        )
        if identity in identities:
            raise ReportingSchemaError(
                "DUPLICATE_METRIC_IDENTITY", f"{row.model} {row.profile} {row.fraction}"
            )
        identities.add(identity)


def _validate_row(row: MetricRow) -> None:
    if not math.isfinite(row.fraction) or row.fraction <= 0.0 or row.fraction > 1.0:
        raise ReportingSchemaError("INVALID_FRACTION", f"{row.fraction}")
    if row.train_stimuli <= 0:
        raise ReportingSchemaError("INVALID_TRAIN_STIMULI", f"{row.train_stimuli}")
    if row.model in SHARED_MODELS and row.profile is not Profile.SHARED_REFERENCE:
        raise ReportingSchemaError("MIXED_REGIME_ROW", row.model)
    if row.model in PROFILED_MODELS and row.profile is Profile.SHARED_REFERENCE:
        raise ReportingSchemaError("MIXED_REGIME_ROW", row.model)
    _validate_required_metrics(row)
    _validate_optional_metrics(row)
    has_rf = row.model in RF_MODELS
    if has_rf and None in (
        row.global_rf,
        row.spatial_rf,
        row.temporal_rf,
        row.exact_cell,
        row.cross_bank_rf,
    ):
        raise ReportingSchemaError("MISSING_VARIATION_FIELD", row.model)
    if row.model in SEEDED_RF_MODELS and row.cross_seed_rf is None:
        raise ReportingSchemaError("MISSING_VARIATION_FIELD", row.model)
    if not has_rf and any(
        value is not None
        for value in (
            row.global_rf,
            row.spatial_rf,
            row.temporal_rf,
            row.exact_cell,
            row.cross_seed_rf,
            row.cross_bank_rf,
        )
    ):
        raise ReportingSchemaError("UNEXPECTED_RF_FIELD", row.model)


def _validate_required_metrics(row: MetricRow) -> None:
    _validate_finite_metric("val_ce", row.val_ce)
    _validate_finite_metric("sampled_nll", row.sampled_nll)
    _validate_finite_metric("bits_per_spike", row.bits_per_spike)


def _validate_optional_metrics(row: MetricRow) -> None:
    _validate_optional_metric("global_rf", row.global_rf)
    _validate_optional_metric("spatial_rf", row.spatial_rf)
    _validate_optional_metric("temporal_rf", row.temporal_rf)
    _validate_optional_metric("exact_cell", row.exact_cell)
    _validate_optional_metric("nearest_type_polarity", row.nearest_type_polarity)
    _validate_optional_metric("prototype_centroid", row.prototype_centroid)
    _validate_optional_metric("cross_seed_rf", row.cross_seed_rf)
    _validate_optional_metric("cross_bank_rf", row.cross_bank_rf)


def _validate_optional_metric(name: str, value: float | None) -> None:
    if value is not None:
        _validate_finite_metric(name, value)


def _validate_finite_metric(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ReportingSchemaError("NONFINITE_METRIC_VALUE", name)


def _aggregate(rows: Sequence[MetricRow]) -> AggregateMetricRow:
    first = rows[0]
    counts = {row.train_stimuli for row in rows}
    params = {row.parameter_count for row in rows}
    if len(counts) != 1 or len(params) != 1:
        raise ReportingSchemaError("MIXED_GROUP_VALUES", first.model)
    return AggregateMetricRow(
        first.fraction,
        first.train_stimuli,
        first.model,
        first.profile,
        first.parameter_count,
        len(rows),
        statistics.fmean(row.val_ce for row in rows),
        _sd(tuple(row.val_ce for row in rows)),
        statistics.fmean(row.sampled_nll for row in rows),
        statistics.fmean(row.bits_per_spike for row in rows),
        _optional_mean(tuple(row.global_rf for row in rows)),
        _optional_sd(tuple(row.global_rf for row in rows)),
        _optional_mean(tuple(row.spatial_rf for row in rows)),
        _optional_mean(tuple(row.temporal_rf for row in rows)),
        _optional_mean(tuple(row.exact_cell for row in rows)),
        _optional_mean(tuple(row.nearest_type_polarity for row in rows)),
        _optional_mean(tuple(row.prototype_centroid for row in rows)),
        _optional_mean(tuple(row.cross_seed_rf for row in rows)),
        _optional_mean(tuple(row.cross_bank_rf for row in rows)),
        sum(row.reuse_status == "reused" for row in rows),
        sum(row.reuse_status != "reused" for row in rows),
    )


def _sort_key(key: _AggregateKey) -> tuple[float, int, str]:
    return (
        key.fraction,
        MODEL_ORDER.index(key.model) if key.model in MODEL_ORDER else len(MODEL_ORDER),
        key.profile.value,
    )


def _sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _optional_mean(values: Sequence[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return statistics.fmean(present) if present else None


def _optional_sd(values: Sequence[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return _sd(present)


__all__ = [
    "AggregateMetricRow",
    "MetricRow",
    "Profile",
    "ReportingSchemaError",
    "aggregate_metric_rows",
    "profile_label",
]
