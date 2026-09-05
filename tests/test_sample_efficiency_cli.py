from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal, assert_never

import pytest

from evaluation.model_comparison.sample_efficiency_reporting import MetricRow, Profile
from evaluation.model_comparison.sample_efficiency_runner import (
    RunnerRequest,
    SampleEfficiencyRunnerError,
    fixture_row_provider,
    run_sample_efficiency,
)
from tests.sample_efficiency_cli_helpers import (
    config,
    counts_by_fraction,
    failing_provider,
    read_rows,
    write_rows,
)


def test_run_sample_efficiency_when_fixture_provider_used_writes_contract_artifacts(
    tmp_path: Path,
) -> None:
    # Given: an isolated sample-efficiency config and a no-training row provider.
    config_path = config(tmp_path)

    # When: the runner composes all three fractions.
    result = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )

    # Then: row counts and 100% reuse/training split match the frozen contract.
    rows = read_rows(result.output_dir / "per-run-metrics.csv")
    assert len(rows) == 153
    assert counts_by_fraction(rows) == {"0.25": 51, "0.5": 51, "1.0": 51}
    assert sum(row["reuse_status"] == "reused" and row["fraction"] == "1.0" for row in rows) == 33
    assert sum(row["reuse_status"] == "trained" and row["fraction"] == "1.0" for row in rows) == 18
    assert {path.name for path in result.output_dir.glob("*.png")} == {
        "sample-efficiency-ce.png",
        "sample-efficiency-rf.png",
    }
    assert (result.output_dir / "identity-manifest.json").stat().st_size > 0
    assert (result.output_dir / "sample-efficiency-results.json").stat().st_size > 0


def test_run_sample_efficiency_when_cache_is_complete_resumes_without_provider_call(
    tmp_path: Path,
) -> None:
    # Given: a completed fixture run with fraction caches.
    config_path = config(tmp_path)
    first = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )

    # When: resume is requested with a provider that would fail if retraining happened.
    second = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, resume=True, row_provider=failing_provider)
    )

    # Then: the final artifact hashes remain stable and no provider call is needed.
    assert second.output_dir == first.output_dir
    assert second.artifact_sha256 == first.artifact_sha256


def test_run_sample_efficiency_when_final_is_complete_resumes_without_fraction_caches(
    tmp_path: Path,
) -> None:
    # Given: a complete final artifact directory but no fraction caches.
    config_path = config(tmp_path)
    first = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )
    shutil.rmtree(tmp_path / "runs")

    # When: resume is requested with a provider that would fail if retraining happened.
    second = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, resume=True, row_provider=failing_provider)
    )

    # Then: the final artifact directory itself is accepted as completed evidence.
    assert second.artifact_sha256 == first.artifact_sha256


FinalCsvCorruption = Literal["truncated", "wrong_profile", "wrong_reuse"]


@pytest.mark.parametrize("corruption", ("truncated", "wrong_profile", "wrong_reuse"))
def test_run_sample_efficiency_when_final_per_run_csv_is_semantically_stale_fails_without_training(
    tmp_path: Path,
    corruption: FinalCsvCorruption,
) -> None:
    # Given: a complete final artifact directory but a schema-valid stale per-run CSV.
    config_path = config(tmp_path)
    first = run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )
    csv_path = first.output_dir / "per-run-metrics.csv"
    match corruption:
        case "truncated":
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            csv_path.write_text(header + "\n", encoding="utf-8")
        case "wrong_profile":
            rows = list(read_rows(csv_path))
            rows[0]["profile"] = Profile.ACTIVE_DOF.value
            write_rows(csv_path, rows)
        case "wrong_reuse":
            rows = list(read_rows(csv_path))
            next(row for row in rows if row["fraction"] == "1.0" and row["reuse_status"] == "reused")["reuse_status"] = "trained"
            write_rows(csv_path, rows)
        case unreachable:
            assert_never(unreachable)

    # When/Then: completed-resume validation rejects the stale final without retraining.
    with pytest.raises(SampleEfficiencyRunnerError, match="STALE_FINAL_OUTPUT"):
        run_sample_efficiency(
            RunnerRequest(Path("."), config_path, resume=True, row_provider=failing_provider)
        )


def test_run_sample_efficiency_when_final_dir_has_pareto_fails_closed(
    tmp_path: Path,
) -> None:
    # Given: a stale partial final directory containing the forbidden canonical figure.
    config_path = config(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "pareto.png").write_bytes(b"stale")

    # When/Then: the runner refuses the directory before promoting new files into it.
    with pytest.raises(SampleEfficiencyRunnerError, match="STALE_FINAL_OUTPUT"):
        run_sample_efficiency(
            RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
        )
    assert {path.name for path in output.iterdir()} == {"pareto.png"}


def test_run_sample_efficiency_when_completed_cache_identity_changes_fails(
    tmp_path: Path,
) -> None:
    # Given: a completed fraction cache whose identity no longer matches the protocol.
    config_path = config(tmp_path)
    run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )
    cache_path = tmp_path / "runs" / "fraction-025" / "metrics-cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["identity"]["train_count"] = 999
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    # When/Then: stale completed state fails closed before new rows are accepted.
    try:
        run_sample_efficiency(
            RunnerRequest(Path("."), config_path, resume=True, row_provider=fixture_row_provider)
        )
    except SampleEfficiencyRunnerError as exc:
        assert exc.code == "STALE_FRACTION_CACHE"
    else:
        raise AssertionError("stale cache unexpectedly resumed")


def test_run_sample_efficiency_when_cache_omits_scientific_identity_fails(
    tmp_path: Path,
) -> None:
    # Given: a completed cache whose identity is missing required scientific provenance.
    config_path = config(tmp_path)
    run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )
    cache_path = tmp_path / "runs" / "fraction-025" / "metrics-cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "candidate0_rf_sha256" in payload["identity"]
    del payload["identity"]["candidate0_rf_sha256"]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    # When/Then: resume rejects the stale cache before accepting cached rows.
    with pytest.raises(SampleEfficiencyRunnerError, match="STALE_FRACTION_CACHE"):
        run_sample_efficiency(
            RunnerRequest(Path("."), config_path, resume=True, row_provider=failing_provider)
        )


def test_run_sample_efficiency_when_cached_row_profile_changes_fails(
    tmp_path: Path,
) -> None:
    # Given: a completed cache whose row payload no longer matches the row contract.
    config_path = config(tmp_path)
    run_sample_efficiency(
        RunnerRequest(Path("."), config_path, row_provider=fixture_row_provider)
    )
    cache_path = tmp_path / "runs" / "fraction-025" / "metrics-cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["rows"][0]["profile"] = "active-dof"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    # When/Then: resume revalidates cached rows and fails closed.
    with pytest.raises(SampleEfficiencyRunnerError, match="STALE_FRACTION_CACHE"):
        run_sample_efficiency(
            RunnerRequest(Path("."), config_path, resume=True, row_provider=failing_provider)
        )
