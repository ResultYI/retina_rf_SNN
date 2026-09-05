from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.model_comparison import sample_efficiency_experiment as experiment
from evaluation.model_comparison.sample_efficiency_runner import fixture_row_provider
from tests.sample_efficiency_cli_helpers import (
    SpyCall,
    call_counts,
    config,
    spy_bank_runner,
    spy_seed_runner,
    wait_for_lock,
)


def test_run_fraction_when_spy_run_set_used_matches_real_training_call_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the real prepared protocol and spy runners installed at the production seam.
    config_path = config(tmp_path)
    prepared = experiment.prepare_sample_efficiency(Path("."), config_path)
    calls: list[SpyCall] = []
    run_set = experiment.SampleEfficiencyRunSet(
        spy_bank_runner(calls, "Bias", None, 16),
        spy_bank_runner(calls, "GLM-SH", None, 7504),
        spy_seed_runner(calls, "LN-LN", 160),
        spy_seed_runner(calls, "Graph-TCN", 144),
        spy_seed_runner(calls, "Mechanistic Retina", 264),
    )
    canonical = tuple(
        row for row in fixture_row_provider(1.0, 112) if row.reuse_status == "reused"
    )
    monkeypatch.setattr(experiment, "canonical_metric_rows", lambda root: canonical)

    # When: the production composition runs a 25% fraction and then the 100% fraction.
    rows_25 = experiment.run_fraction(
        Path("."), prepared, prepared.protocol.subsets[0], run_set, lambda event: None
    )
    calls_25 = tuple(calls)
    calls.clear()
    rows_100 = experiment.run_fraction(
        Path("."), prepared, prepared.protocol.subsets[2], run_set, lambda event: None
    )

    # Then: 25/50-style fractions train 51 rows; 100% reuses 33 and trains only active controls.
    assert len(rows_25) == 51
    assert call_counts(calls_25) == {
        ("Bias", 264): 3,
        ("GLM-SH", 264): 3,
        ("LN-LN", 264): 9,
        ("Graph-TCN", 264): 9,
        ("Mechanistic Retina", 264): 9,
        ("LN-LN", 136): 9,
        ("Graph-TCN", 136): 9,
    }
    assert len(rows_100) == 51
    assert sum(row.reuse_status == "reused" for row in rows_100) == 33
    assert call_counts(calls) == {("LN-LN", 136): 9, ("Graph-TCN", 136): 9}
    assert {item[3] for item in calls_25} == {28}
    assert len({item[4] for item in calls_25}) == 1
    assert len({item[5] for item in calls_25}) == 1


def test_cli_when_second_fixture_worker_starts_reports_run_locked(
    tmp_path: Path,
) -> None:
    # Given: one fixture worker holding the exclusive run lock.
    config_path = config(tmp_path)
    first = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "scripts/run_sample_efficiency.py",
            "--config",
            str(config_path),
            "--fixture-mode",
            "--hold-lock-seconds",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wait_for_lock(tmp_path / "runs" / ".sample-efficiency.lock")

    # When: a second worker targets the same run directory.
    second = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/run_sample_efficiency.py",
            "--config",
            str(config_path),
            "--fixture-mode",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    first_stdout, first_stderr = first.communicate(timeout=30)

    # Then: the second process is rejected with the stable code.
    assert second.returncode != 0
    assert "RUN_LOCKED" in second.stderr
    assert first.returncode == 0, first_stdout + first_stderr


def test_cli_validate_only_prints_frozen_contract_without_training(
    tmp_path: Path,
) -> None:
    # Given: an isolated config pointing at the canonical T=2 benchmark.
    config_path = config(tmp_path)

    # When: validate-only is invoked through the real script surface.
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "scripts/run_sample_efficiency.py",
            "--config",
            str(config_path),
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Then: it reports identity/count/profile contracts without training.
    payload = json.loads(result.stdout)
    assert payload["fractions"] == [0.25, 0.5, 1.0]
    assert payload["train_counts"] == [28, 56, 112]
    assert payload["mechanistic_parameters"] == {
        "total": 264,
        "requires_grad": 264,
        "optimizer_listed": 136,
        "nonzero_gradient": None,
        "actually_updated": None,
    }
    assert payload["training_rows_by_fraction"] == {"0.25": 51, "0.5": 51, "1.0": 18}
