from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evaluation.mechanistic_retina.artifacts import FINAL_TEST_BOUNDARY
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.artifacts import write_csv, write_json
from evaluation.model_comparison.sample_efficiency_experiment import (
    DEFAULT_RUN_SET,
    SampleEfficiencyRunSet,
    prepare_sample_efficiency,
    run_fraction,
    validation_contract,
)
from evaluation.model_comparison.sample_efficiency_presentation import (
    build_decision_payload,
    sample_efficiency_report,
    write_sample_efficiency_figures,
)
from evaluation.model_comparison.sample_efficiency_protocol import (
    SampleEfficiencyProtocol,
    SampleEfficiencySubset,
    load_sample_efficiency_protocol,
)
from evaluation.model_comparison.sample_efficiency_reporting import (
    AggregateMetricRow,
    MetricRow,
    Profile,
    aggregate_metric_rows,
)
from evaluation.model_comparison.sample_efficiency_rows import aggregate_fields, aggregate_payload, comparison_rows, fixture_row_provider, parse_row, row_fields, row_payload
from evaluation.model_comparison.sample_efficiency_state import CacheRequest, ExclusiveLock, FinalArtifactRequest, SampleEfficiencyStateError, artifact_hashes, cache_path, completed_final, prepare_staging, promote_staging, read_cache, validate_final_output, write_cache
from evaluation.model_comparison.types import ProgressEvent


FractionRowProvider = Callable[[float, int], tuple[MetricRow, ...]]
REQUIRED_ARTIFACTS: Final = ("identity-manifest.json", "experiment-config.yaml", "parameter-counts.json", "sample-efficiency-results.json", "per-run-metrics.csv", "model-comparison.csv", "decision-report-zh.md", "sample-efficiency-rf.png", "sample-efficiency-ce.png")
_EXPECTED_TRAINED: Final = {("Bias", Profile.SHARED_REFERENCE, "trained"): 3, ("GLM-SH", Profile.SHARED_REFERENCE, "trained"): 3, ("LN-LN", Profile.ARCHITECTURE_SIZE, "trained"): 9, ("Graph-TCN", Profile.ARCHITECTURE_SIZE, "trained"): 9, ("Mechanistic Retina", Profile.ARCHITECTURE_SIZE, "trained"): 9, ("LN-LN", Profile.ACTIVE_DOF, "trained"): 9, ("Graph-TCN", Profile.ACTIVE_DOF, "trained"): 9}
_EXPECTED_100: Final = {("Bias", Profile.SHARED_REFERENCE, "reused"): 3, ("GLM-SH", Profile.SHARED_REFERENCE, "reused"): 3, ("LN-LN", Profile.ARCHITECTURE_SIZE, "reused"): 9, ("Graph-TCN", Profile.ARCHITECTURE_SIZE, "reused"): 9, ("Mechanistic Retina", Profile.ARCHITECTURE_SIZE, "reused"): 9, ("LN-LN", Profile.ACTIVE_DOF, "trained"): 9, ("Graph-TCN", Profile.ACTIVE_DOF, "trained"): 9}


@dataclass(frozen=True, slots=True)
class RunnerRequest:
    root: Path
    config_path: Path
    resume: bool = False
    row_provider: FractionRowProvider | None = None
    run_set: SampleEfficiencyRunSet = DEFAULT_RUN_SET
    hold_lock_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RunnerResult:
    output_dir: Path
    artifact_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SampleEfficiencyRunnerError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def run_sample_efficiency(request: RunnerRequest) -> RunnerResult:
    try:
        return _run_sample_efficiency(request)
    except SampleEfficiencyStateError as exc:
        raise SampleEfficiencyRunnerError(exc.code, exc.detail) from exc


def _run_sample_efficiency(request: RunnerRequest) -> RunnerResult:
    protocol = load_sample_efficiency_protocol(request.config_path)
    root = request.root
    run_root = _resolve(root, protocol.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    with ExclusiveLock(run_root / ".sample-efficiency.lock"):
        if request.hold_lock_seconds > 0.0:
            time.sleep(request.hold_lock_seconds)
        contract = build_validation_contract(root, request.config_path)
        output = _resolve(root, protocol.output_dir)
        final_is_current = False
        if request.resume and output.exists():
            try:
                completed = completed_final(FinalArtifactRequest(output, REQUIRED_ARTIFACTS, _identity(protocol, 153, contract)))
            except SampleEfficiencyStateError:
                _assert_completed_final_rows(protocol, output)
            else:
                _assert_completed_final_rows(protocol, output)
                final_is_current = True
                if not any(cache_path(root, protocol.run_dir, subset.fraction).exists() for subset in protocol.subsets):
                    return RunnerResult(completed.output_dir, completed.artifact_sha256)
        if not request.resume and output.exists():
            raise SampleEfficiencyRunnerError("STALE_FINAL_OUTPUT", str(output))
        rows = _load_or_run_rows(root, protocol, request, contract)
        if request.resume and _resolve(root, protocol.output_dir).exists():
            if not final_is_current:
                output = _write_final_artifacts(root, request.config_path, protocol, rows, contract, replace_existing=True)
                return RunnerResult(output, artifact_hashes(output, REQUIRED_ARTIFACTS))
            validate_final_output(FinalArtifactRequest(output, REQUIRED_ARTIFACTS, _identity(protocol, len(rows), contract)))
            return RunnerResult(output, artifact_hashes(output, REQUIRED_ARTIFACTS))
        output = _write_final_artifacts(root, request.config_path, protocol, rows, contract)
        return RunnerResult(output, artifact_hashes(output, REQUIRED_ARTIFACTS))


def build_validation_contract(root: Path, config_path: Path) -> Mapping[str, JsonValue]:
    return validation_contract(root, config_path)


def _load_or_run_rows(
    root: Path,
    protocol: SampleEfficiencyProtocol,
    request: RunnerRequest,
    contract: Mapping[str, JsonValue],
) -> tuple[MetricRow, ...]:
    prepared = None if request.row_provider is not None else prepare_sample_efficiency(root, request.config_path)
    rows: list[MetricRow] = []
    for subset in protocol.subsets:
        identity = _fraction_identity(request.config_path, subset, contract)
        cache = cache_path(root, protocol.run_dir, subset.fraction)
        cached = _read_cache(cache, identity, subset)
        if cached is not None:
            rows.extend(cached)
            continue
        if request.resume and cache.exists():
            raise SampleEfficiencyRunnerError("STALE_FRACTION_CACHE", str(cache))
        if request.row_provider is None:
            if prepared is None:
                raise SampleEfficiencyRunnerError("INTERNAL_PROVIDER_MISSING", str(subset.fraction))
            new_rows = run_fraction(root, prepared, subset, request.run_set, _progress)
        else:
            new_rows = request.row_provider(subset.fraction, subset.train_count)
        _assert_fraction_rows(subset, new_rows)
        write_cache(CacheRequest(cache, identity), tuple(row_payload(row) for row in new_rows))
        rows.extend(new_rows)
    return tuple(rows)


def _write_final_artifacts(
    root: Path,
    config_path: Path,
    protocol: SampleEfficiencyProtocol,
    rows: Sequence[MetricRow],
    contract: Mapping[str, JsonValue],
    replace_existing: bool = False,
) -> Path:
    output = _resolve(root, protocol.output_dir)
    if output.exists() and not replace_existing:
        raise SampleEfficiencyRunnerError("STALE_FINAL_OUTPUT", str(output))
    staging = prepare_staging(output)
    report_rows = comparison_rows(rows)
    aggregates = aggregate_metric_rows(report_rows)
    decision = build_decision_payload(aggregates)
    write_json(staging / "identity-manifest.json", _identity(protocol, len(rows), contract))
    staging.joinpath("experiment-config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(staging / "parameter-counts.json", build_validation_contract(root, config_path)["mechanistic_parameters"])
    write_json(staging / "sample-efficiency-results.json", _results_payload(rows, aggregates, decision))
    write_csv(staging / "per-run-metrics.csv", row_fields(), tuple(row_payload(row) for row in rows))
    write_csv(staging / "model-comparison.csv", aggregate_fields(), tuple(aggregate_payload(row) for row in aggregates))
    staging.joinpath("decision-report-zh.md").write_text(
        sample_efficiency_report(aggregates, decision), encoding="utf-8"
    )
    write_sample_efficiency_figures(staging, report_rows)
    promote_staging(staging, FinalArtifactRequest(output, REQUIRED_ARTIFACTS, _identity(protocol, len(rows), contract)), replace_existing)
    return output


def _read_cache(
    path: Path,
    identity: Mapping[str, JsonValue],
    subset: SampleEfficiencySubset,
) -> tuple[MetricRow, ...] | None:
    cached = read_cache(CacheRequest(path, identity))
    if cached is None:
        return None
    try:
        parsed = tuple(parse_row(row) for row in cached)
        _assert_fraction_rows(subset, parsed)
        return parsed
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleEfficiencyRunnerError("STALE_FRACTION_CACHE", str(path))
    except SampleEfficiencyRunnerError as exc:
        raise SampleEfficiencyRunnerError("STALE_FRACTION_CACHE", str(path)) from exc


def _assert_completed_final_rows(protocol: SampleEfficiencyProtocol, output: Path) -> None:
    try:
        with (output / "per-run-metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = tuple(
                parse_row(
                    {key: None if key == "model_seed" and value == "" else value for key, value in row.items()}
                )
                for row in csv.DictReader(handle)
            )
        for subset in protocol.subsets:
            _assert_fraction_rows(subset, tuple(row for row in rows if row.fraction == subset.fraction))
    except (KeyError, OSError, TypeError, ValueError, SampleEfficiencyRunnerError) as exc:
        raise SampleEfficiencyRunnerError("STALE_FINAL_OUTPUT", str(output)) from exc


def _assert_fraction_rows(subset: SampleEfficiencySubset, rows: Sequence[MetricRow]) -> None:
    if len(rows) != 51 or {row.fraction for row in rows} != {subset.fraction}:
        raise SampleEfficiencyRunnerError("FRACTION_ROW_CONTRACT", str(subset.fraction))
    if {row.train_stimuli for row in rows} != {subset.train_count}:
        raise SampleEfficiencyRunnerError("FRACTION_ROW_CONTRACT", str(subset.train_count))
    if _row_distribution(rows) != _expected_distribution(subset.fraction):
        raise SampleEfficiencyRunnerError("FRACTION_ROW_CONTRACT", "model/profile distribution")
    if subset.fraction == 1.0:
        reused = sum(row.reuse_status == "reused" for row in rows)
        trained = sum(row.reuse_status == "trained" for row in rows)
        if (reused, trained) != (33, 18):
            raise SampleEfficiencyRunnerError("CANONICAL_REUSE_CONTRACT", f"{reused}/{trained}")


def _row_distribution(rows: Sequence[MetricRow]) -> Mapping[tuple[str, Profile, str], int]:
    keys = {(row.model, row.profile, row.reuse_status) for row in rows}
    return {key: sum((row.model, row.profile, row.reuse_status) == key for row in rows) for key in keys}


def _expected_distribution(fraction: float) -> Mapping[tuple[str, Profile, str], int]:
    return _EXPECTED_TRAINED if fraction != 1.0 else _EXPECTED_100


def _fraction_identity(
    config_path: Path,
    subset: SampleEfficiencySubset,
    contract: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    fraction_hashes = contract["fraction_hashes"]
    if not isinstance(fraction_hashes, dict):
        raise SampleEfficiencyRunnerError("FRACTION_IDENTITY_CONTRACT", str(subset.fraction))
    return {
        **contract,
        "lineage": "candidate0-t2-sample-efficiency-active-dof-v1",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "fraction": subset.fraction,
        "train_count": subset.train_count,
        "indices": list(subset.indices),
        "fraction_identity": fraction_hashes[str(subset.fraction)],
    }


def _identity(
    protocol: SampleEfficiencyProtocol,
    row_count: int,
    contract: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        **contract,
        "lineage": "candidate0-t2-sample-efficiency-active-dof-v1",
        "fractions": [subset.fraction for subset in protocol.subsets],
        "train_counts": [subset.train_count for subset in protocol.subsets],
        "selection_seed": protocol.selection_seed,
        "row_count": row_count,
        "final_test_boundary": list(FINAL_TEST_BOUNDARY),
    }


def _results_payload(
    rows: Sequence[MetricRow],
    aggregates: Sequence[AggregateMetricRow],
    decision: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        "row_count": len(rows),
        "rows_by_fraction": {str(fraction): sum(row.fraction == fraction for row in rows) for fraction in (0.25, 0.5, 1.0)},
        "reused_100_fraction_rows": sum(row.fraction == 1.0 and row.reuse_status == "reused" for row in rows),
        "trained_100_fraction_rows": sum(row.fraction == 1.0 and row.reuse_status == "trained" for row in rows),
        "aggregates": [aggregate_payload(row) for row in aggregates],
        "decision": decision,
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _progress(event: ProgressEvent) -> None:
    print(json.dumps({"model": event.model, "bank": event.bank_seed, "seed": event.model_seed, "step": event.step, "loss": event.loss}), flush=True)

__all__ = [
    "REQUIRED_ARTIFACTS",
    "RunnerRequest",
    "RunnerResult",
    "SampleEfficiencyRunnerError",
    "build_validation_contract",
    "fixture_row_provider",
    "run_sample_efficiency",
]
