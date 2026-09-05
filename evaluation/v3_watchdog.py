from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


@dataclass(frozen=True, slots=True)
class WatchdogRequest:
    command: tuple[str, ...]
    workdir: Path
    log_path: Path
    monitor_path: Path
    failure_path: Path
    progress_artifact: Path
    monitor_interval_seconds: float = 300.0
    poll_interval_seconds: float = 1.0
    stall_intervals: int = 2


@dataclass(frozen=True, slots=True)
class WatchdogResult:
    exit_code: int
    status: str
    pid: int


@dataclass(frozen=True, slots=True)
class ProgressState:
    gate: str
    seed: int | None
    method: str | None
    bank: int | None
    condition: str
    step: int
    metric: float | None


_ERROR_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|RuntimeError:|MemoryError:|"
    r"FileNotFoundError:|PermissionError:|OSError:|out[- ]of[- ]memory|"
    r"access violation",
    re.IGNORECASE,
)
_SOURCE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')
_EXCEPTION_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", re.MULTILINE)


def run_watchdog(request: WatchdogRequest) -> WatchdogResult:
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    request.monitor_path.parent.mkdir(parents=True, exist_ok=True)
    request.failure_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    with request.log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        with subprocess.Popen(
            request.command,
            cwd=request.workdir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        ) as process:
            progress = ProgressState("startup", None, None, None, "gate1", 0, None)
            last_offset = 0
            last_signature = None
            unchanged = 0
            next_monitor = time.monotonic() + request.monitor_interval_seconds
            _append_monitor(request, process.pid, process.poll(), progress, "STARTED")
            while True:
                exit_code = process.poll()
                text, last_offset = _new_log_text(request.log_path, last_offset)
                parsed = _latest_progress(text)
                if parsed is not None:
                    progress = parsed
                    if last_signature is None:
                        last_signature = _progress_signature(request, progress)
                failure_reason = _immediate_failure(text, progress, exit_code)
                if failure_reason is not None:
                    _terminate(process)
                    final_exit = process.poll()
                    code = 1 if final_exit in {None, 0} else int(final_exit)
                    _write_failure(request, process.pid, code, failure_reason, progress)
                    _append_monitor(request, process.pid, code, progress, "FAILED")
                    return WatchdogResult(code, failure_reason, process.pid)
                if exit_code is not None:
                    status = "COMPLETED" if exit_code == 0 else "NON_ZERO_EXIT"
                    if exit_code != 0:
                        _write_failure(request, process.pid, exit_code, status, progress)
                    _append_monitor(request, process.pid, exit_code, progress, status)
                    return WatchdogResult(exit_code, status, process.pid)
                now = time.monotonic()
                if now >= next_monitor:
                    signature = _progress_signature(request, progress)
                    unchanged = unchanged + 1 if signature == last_signature else 0
                    last_signature = signature
                    _append_monitor(request, process.pid, None, progress, "RUNNING")
                    if unchanged >= request.stall_intervals:
                        _terminate(process)
                        code = process.poll() or 1
                        _write_failure(request, process.pid, code, "STALLED", progress)
                        _append_monitor(request, process.pid, code, progress, "STALLED")
                        return WatchdogResult(code, "STALLED", process.pid)
                    next_monitor = now + request.monitor_interval_seconds
                time.sleep(request.poll_interval_seconds)


def _latest_progress(text: str) -> ProgressState | None:
    for line in reversed(text.splitlines()):
        if not line.startswith("PROGRESS "):
            continue
        try:
            payload = json.loads(line.removeprefix("PROGRESS "))
            metric_value = payload.get("metric")
            metric = None if metric_value is None else float(metric_value)
            return ProgressState(
                str(payload.get("gate", "unknown")),
                None if payload.get("seed") is None else int(payload["seed"]),
                None if payload.get("method") is None else str(payload["method"]),
                None if payload.get("bank") is None else int(payload["bank"]),
                str(payload.get("condition", "unknown")),
                int(payload.get("step", 0)),
                metric,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return ProgressState("invalid-progress", None, None, None, "unknown", 0, math.nan)
    return None


def _immediate_failure(
    text: str,
    progress: ProgressState,
    exit_code: int | None,
) -> str | None:
    if progress.metric is not None and not math.isfinite(progress.metric):
        return "NON_FINITE_METRIC"
    if _ERROR_PATTERN.search(text):
        return "PYTHON_RUNTIME_ERROR"
    if exit_code is not None and exit_code != 0:
        return "NON_ZERO_EXIT"
    return None


def _new_log_text(path: Path, offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", offset
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        return text, handle.tell()


def _progress_signature(request: WatchdogRequest, progress: ProgressState):
    artifact_mtime = (
        request.progress_artifact.stat().st_mtime_ns
        if request.progress_artifact.exists()
        else None
    )
    log_size = request.log_path.stat().st_size if request.log_path.exists() else 0
    return progress.method, progress.step, progress.metric, log_size, artifact_mtime


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _append_monitor(
    request: WatchdogRequest,
    pid: int,
    exit_code: int | None,
    progress: ProgressState,
    status: str,
) -> None:
    entry = {
        "timestamp": _timestamp(),
        "pid": pid,
        "alive": exit_code is None,
        "exit_code": exit_code,
        "status": status,
        "gate": progress.gate,
        "seed": progress.seed,
        "method": progress.method,
        "bank": progress.bank,
        "condition": progress.condition,
        "step": progress.step,
        "last_finite_metric": progress.metric
        if progress.metric is not None and math.isfinite(progress.metric)
        else None,
        "log_size": request.log_path.stat().st_size if request.log_path.exists() else 0,
        "progress_mtime_ns": request.progress_artifact.stat().st_mtime_ns
        if request.progress_artifact.exists()
        else None,
    }
    with request.monitor_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")


def _write_failure(
    request: WatchdogRequest,
    pid: int,
    exit_code: int,
    reason: str,
    progress: ProgressState,
) -> None:
    lines = request.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-100:]
    joined = "\n".join(tail)
    sources = _SOURCE_PATTERN.findall(joined)
    exceptions = _EXCEPTION_PATTERN.findall(joined)
    payload = {
        "timestamp": _timestamp(),
        "command": list(request.command),
        "pid": pid,
        "gate": progress.gate,
        "seed": progress.seed,
        "method": progress.method,
        "bank": progress.bank,
        "condition": progress.condition,
        "last_completed_step": progress.step,
        "last_finite_metric": progress.metric
        if progress.metric is not None and math.isfinite(progress.metric)
        else None,
        "exit_code": exit_code,
        "exception_type": exceptions[-1] if exceptions else reason,
        "source_file": sources[-1][0] if sources else None,
        "source_line": int(sources[-1][1]) if sources else None,
        "last_100_relevant_log_lines": tail,
        "latest_valid_progress_artifact": str(request.progress_artifact)
        if request.progress_artifact.exists()
        else None,
    }
    request.failure_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["WatchdogRequest", "WatchdogResult", "run_watchdog"]
