from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from evaluation.model_comparison.config import load_comparison_config
from evaluation.v3_watchdog import WatchdogRequest


REQUIRED_EVIDENCE = (
    "identity-manifest.json",
    "experiment-config.yaml",
    "parameter-counts.json",
    "prediction-results.json",
    "rf-results.json",
    "stability-results.json",
    "per-run-metrics.csv",
    "per-cell-metrics.csv",
    "model-comparison.csv",
    "pareto.png",
    "runtime-monitor.jsonl",
    "commands.json",
    "decision-report-zh.md",
)


@dataclass(frozen=True, slots=True)
class RunnerContract:
    watchdog: WatchdogRequest
    required_evidence: tuple[str, ...]
    controlled_retries: int


def runner_contract(root: Path) -> RunnerContract:
    config_path = root / "configs/model_comparison_t2.yaml"
    config = load_comparison_config(config_path)
    output = root / config.output_dir
    run_dir = root / config.run_dir
    command = (
        sys.executable,
        "-u",
        str(root / "scripts/run_model_comparison.py"),
        "--worker",
    )
    watchdog = WatchdogRequest(
        command,
        root,
        run_dir / "model-comparison.log",
        output / "runtime-monitor.jsonl",
        output / "failure.json",
        run_dir / "progress.json",
        config.monitor_interval_seconds,
        1.0,
        config.stall_intervals,
    )
    return RunnerContract(watchdog, REQUIRED_EVIDENCE, 1)


__all__ = ["REQUIRED_EVIDENCE", "RunnerContract", "runner_contract"]
