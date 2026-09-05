#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/run_mechanism_checkpoint_replay.py
# 3. Or make executable and run:
#      chmod +x scripts/run_mechanism_checkpoint_replay.py && ./scripts/run_mechanism_checkpoint_replay.py
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.artifacts import write_json
from evaluation.mechanistic_retina.mechanism_replay_protocol import (
    ReplayProtocolRequest,
    run_checkpoint_replay,
)
from evaluation.mechanistic_retina.mechanism_run_types import ProgressEvent
from evaluation.v3_watchdog import WatchdogRequest, run_watchdog


_EVIDENCE = _REPO_ROOT / ".omo/evidence/mechanism-heldout-final"
_CHECKPOINTS = _REPO_ROOT / "runs/mechanism_identifiable_final"
_OUTPUT_NAMES = (
    "identity-manifest.json",
    "replay-results.json",
    "checkpoint-manifest.json",
    "heldout-h1-results.json",
    "heldout-ac-results.json",
    "false-positive-results.json",
    "per-seed-metrics.csv",
    "runtime-monitor.jsonl",
    "commands.json",
    "decision-report-zh.md",
    "failure.json",
)


@dataclass(frozen=True, slots=True)
class RunnerContract:
    monitor_interval_seconds: float
    stall_intervals: int
    controlled_retries: int
    required_checkpoints: int
    heldout_optimizer_steps: int
    output_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunnerError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def runner_contract() -> RunnerContract:
    return RunnerContract(300.0, 2, 1, 15, 0, _OUTPUT_NAMES)


def _progress(event: ProgressEvent) -> None:
    print(
        "PROGRESS "
        + json.dumps(
            {
                "gate": event.phase,
                "phase": event.phase,
                "teacher": event.teacher,
                "method": event.model,
                "structural_variant": event.model,
                "seed": event.seed,
                "step": event.step,
                "train_or_validation_ce": event.ce,
                "CE": event.ce,
                "RF": event.rf,
                "pathway_gate": event.gate,
                "gate_finite": True,
                "rf_finite": True,
                "heartbeat": True,
                "metric": event.ce,
                "condition": event.teacher,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _worker(run_id: str) -> None:
    run_checkpoint_replay(ReplayProtocolRequest(_REPO_ROOT, run_id, _progress))


def _supervised() -> int:
    _require_fresh(_EVIDENCE, "evidence")
    _require_fresh(_CHECKPOINTS, "checkpoint")
    _EVIDENCE.mkdir(parents=True, exist_ok=True)
    _CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    _EVIDENCE.joinpath("runtime-monitor.jsonl").write_text("", encoding="utf-8")
    run_id = (
        "mechanism-heldout-final-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{uuid4().hex[:8]}"
    )
    command = (
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        run_id,
    )
    write_json(
        _EVIDENCE / "commands.json",
        {
            "run_id": run_id,
            "command": list(command),
            "controlled_retry_limit": 1,
            "controlled_retries_used": 0,
            "heldout_optimizer_steps": 0,
            "minimum_checkpoint_count": 15,
        },
    )
    runtime_log = _CHECKPOINTS / ".runtime.log"
    result = run_watchdog(
        WatchdogRequest(
            command,
            _REPO_ROOT,
            runtime_log,
            _EVIDENCE / "runtime-monitor.jsonl",
            _EVIDENCE / "failure.json",
            _EVIDENCE / "checkpoint-manifest.json",
            300.0,
            1.0,
            2,
        )
    )
    runtime_log.unlink(missing_ok=True)
    return result.exit_code


def _require_fresh(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise RunnerError(f"{label} output is not fresh: {path}")


def _main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
        return 0
    if len(sys.argv) != 1:
        raise RunnerError("usage: run_mechanism_checkpoint_replay.py")
    return _supervised()


if __name__ == "__main__":
    raise SystemExit(_main())
