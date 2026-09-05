from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from evaluation.v3_watchdog import WatchdogRequest, run_watchdog


def test_v3_watchdog_module_exists() -> None:
    # Given / When
    specification = importlib.util.find_spec("evaluation.v3_watchdog")

    # Then
    assert specification is not None


def test_watchdog_records_non_zero_exit(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "nonzero"
    root.mkdir()
    command = (sys.executable, "-u", "-c", "import sys; sys.exit(3)")

    # When
    result = run_watchdog(_request(root, command))

    # Then
    assert result.exit_code == 3
    assert result.status == "NON_ZERO_EXIT"
    assert json.loads((root / "failure.json").read_text(encoding="utf-8"))["exit_code"] == 3


def test_watchdog_rejects_non_finite_progress_metric(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "nan"
    root.mkdir()
    progress = json.dumps(
        {"gate": "gate1", "seed": 19, "condition": "112x8", "step": 25, "metric": "nan"}
    )
    command = (sys.executable, "-u", "-c", f"print('PROGRESS {progress}', flush=True)")

    # When
    result = run_watchdog(_request(root, command))

    # Then
    assert result.status == "NON_FINITE_METRIC"
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["last_finite_metric"] is None


def test_watchdog_terminates_stalled_process_after_two_intervals(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "stalled"
    root.mkdir()
    progress = json.dumps(
        {"gate": "gate1", "seed": 19, "condition": "112x8", "step": 25, "metric": 0.5}
    )
    command = (
        sys.executable,
        "-u",
        "-c",
        f"import time; print('PROGRESS {progress}', flush=True); time.sleep(5)",
    )

    # When
    result = run_watchdog(_request(root, command))

    # Then
    assert result.status == "STALLED"
    assert (root / "failure.json").exists()


def test_watchdog_records_current_method_in_monitor(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "method"
    root.mkdir()
    progress = json.dumps(
        {
            "gate": "noise-free-gate1",
            "seed": 19,
            "method": "v4-stage05",
            "condition": "112x8",
            "step": 50,
            "metric": 0.25,
        }
    )
    command = (sys.executable, "-u", "-c", f"print('PROGRESS {progress}', flush=True)")

    # When
    result = run_watchdog(_request(root, command))

    # Then
    records = [
        json.loads(line)
        for line in (root / "runtime-monitor.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result.status == "COMPLETED"
    assert records[-1]["method"] == "v4-stage05"


def test_watchdog_new_run_ignores_previous_failure_log(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "retry"
    root.mkdir()
    failed = (sys.executable, "-u", "-c", "raise RuntimeError('first run')")
    succeeded = (sys.executable, "-u", "-c", "print('fresh run', flush=True)")
    first = run_watchdog(_request(root, failed))

    # When
    second = run_watchdog(_request(root, succeeded))

    # Then
    assert first.status == "PYTHON_RUNTIME_ERROR"
    assert second.status == "COMPLETED"
    assert (root / "run.log").read_text(encoding="utf-8").strip() == "fresh run"


def _request(root: Path, command: tuple[str, ...]) -> WatchdogRequest:
    return WatchdogRequest(
        command,
        Path.cwd(),
        root / "run.log",
        root / "runtime-monitor.jsonl",
        root / "failure.json",
        root / "gate1-results.json",
        monitor_interval_seconds=0.05,
        poll_interval_seconds=0.01,
        stall_intervals=2,
    )
