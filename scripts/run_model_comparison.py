#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "matplotlib", "numpy", "torch"]
# ///

# Usage:
#   D:\anaconda\python.exe scripts\run_model_comparison.py

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.model_comparison.artifacts import validate_artifact_set, write_json
from evaluation.model_comparison.experiment import run_experiment
from evaluation.model_comparison.runner_support import runner_contract
from evaluation.model_comparison.types import ProgressEvent
from evaluation.v3_watchdog import run_watchdog


CONFIG_PATH = ROOT / "configs/model_comparison_t2.yaml"


def _progress(event: ProgressEvent) -> None:
    payload = {
        "gate": "canonical-model-comparison-t2",
        "method": event.model,
        "model": event.model,
        "bank": event.bank_seed,
        "seed": event.model_seed,
        "step": event.step,
        "iteration": event.step,
        "loss": event.loss,
        "metric": event.loss,
        "finite_metrics": True,
        "heartbeat": datetime.now(timezone.utc).isoformat(),
        "condition": "T=2",
    }
    progress_path = ROOT / "runs/canonical_model_comparison_t2/progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(progress_path, payload)
    print("PROGRESS " + json.dumps(payload, allow_nan=False, sort_keys=True), flush=True)


def _worker() -> None:
    run_experiment(ROOT, CONFIG_PATH, _progress)


def _supervised() -> int:
    contract = runner_contract(ROOT)
    output = contract.watchdog.monitor_path.parent
    contract.watchdog.log_path.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    contract.watchdog.monitor_path.write_text("", encoding="utf-8")
    write_json(
        output / "commands.json",
        {
            "run_id": "canonical-model-comparison-t2-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8],
            "supervisor_command": [sys.executable, str(Path(__file__).resolve())],
            "worker_command": list(contract.watchdog.command),
            "environment": {
                "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1",
            },
            "monitor_interval_seconds": contract.watchdog.monitor_interval_seconds,
            "stall_intervals": contract.watchdog.stall_intervals,
            "controlled_retry_limit": contract.controlled_retries,
            "controlled_retries_used": 0,
            "verification_commands": [
                [sys.executable, "-m", "pytest", "-q", "tests/test_model_comparison.py", "tests/test_v3_watchdog.py"],
                [sys.executable, "-m", "compileall", "-q", "baselines/lnln_subunit.py", "baselines/graph_tcn.py", "evaluation/model_comparison", "scripts/run_model_comparison.py", "tests/test_model_comparison.py"],
            ],
        },
    )
    result = run_watchdog(contract.watchdog)
    if result.exit_code == 0:
        validate_artifact_set(output, contract.required_evidence)
    return result.exit_code


def _main() -> int:
    if sys.argv[1:] == ["--worker"]:
        _worker()
        return 0
    if sys.argv[1:]:
        raise SystemExit("usage: run_model_comparison.py")
    return _supervised()


if __name__ == "__main__":
    raise SystemExit(_main())
