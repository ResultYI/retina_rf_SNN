#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/run_mechanism_identifiability.py
# 3. Or make executable and run:
#      chmod +x scripts/run_mechanism_identifiability.py && ./scripts/run_mechanism_identifiability.py
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
from evaluation.mechanistic_retina.mechanism_identifiability import load_mechanism_config
from evaluation.mechanistic_retina.mechanism_protocol import ProtocolRequest, run_protocol
from evaluation.mechanistic_retina.mechanism_run_types import ProgressEvent
from evaluation.v3_watchdog import WatchdogRequest, run_watchdog


_CONFIG = _REPO_ROOT / "configs/mechanism_identifiability.yaml"
_OUTPUT_NAMES = (
    "identity-manifest.json",
    "experiment-config.yaml",
    "diagnosis-results.json",
    "teacher-preflight-results.json",
    "noise-free-results.json",
    "sampled-confirmation-results.json",
    "per-pathway-metrics.csv",
    "per-cell-metrics.csv",
    "runtime-monitor.jsonl",
    "run.log",
    "failure.json",
    "commands.json",
    "decision-report-zh.md",
)
@dataclass(frozen=True, slots=True)
class RunnerContract:
    monitor_interval_seconds: float
    stall_intervals: int
    controlled_retries: int
    output_names: tuple[str, ...]


def runner_contract() -> RunnerContract:
    return RunnerContract(300.0, 2, 1, _OUTPUT_NAMES)


def _progress(event: ProgressEvent) -> None:
    print(
        "PROGRESS "
        + json.dumps(
            {
                "gate": event.phase,
                "phase": event.phase,
                "teacher": event.teacher,
                "method": event.model,
                "seed": event.seed,
                "step": event.step,
                "CE": event.ce,
                "RF": event.rf,
                "pathway_gate": event.gate,
                "metric": event.ce,
                "condition": event.teacher,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _worker() -> None:
    run_protocol(ProtocolRequest(_REPO_ROOT, _CONFIG, _progress))


def _supervised() -> int:
    config = load_mechanism_config(_CONFIG)
    output = _REPO_ROOT / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("runtime-monitor.jsonl").write_text("", encoding="utf-8")
    command = (sys.executable, "-u", str(Path(__file__).resolve()), "--worker")
    write_json(
        output / "commands.json",
        {
            "run_id": f"mechanism-identifiable-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
            "command": list(command),
            "controlled_retry_limit": 1,
        },
    )
    result = run_watchdog(
        WatchdogRequest(
            command,
            _REPO_ROOT,
            output / "run.log",
            output / "runtime-monitor.jsonl",
            output / "failure.json",
            output / "diagnosis-results.json",
            config.monitor_interval_seconds,
            1.0,
            config.stall_intervals,
        )
    )
    return result.exit_code


def _main() -> int:
    if sys.argv[1:] == ["--worker"]:
        _worker()
        return 0
    if sys.argv[1:]:
        raise SystemExit("usage: run_mechanism_identifiability.py")
    return _supervised()


if __name__ == "__main__":
    raise SystemExit(_main())
