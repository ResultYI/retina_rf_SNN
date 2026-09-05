from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from evaluation.mechanistic_retina.rf_base import CandidateTeacherUsage


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    candidate0_path: Path
    candidate_teacher_usage: CandidateTeacherUsage
    candidate_teacher_reference_index: int | None
    output_dir: Path
    run_dir: Path
    data_seed: int
    bank_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    trials: int
    steps: int
    checkpoints: tuple[int, ...]
    learning_rate: float
    batch_size: int
    monitor_interval_seconds: float
    stall_intervals: int
    candidate0_rf_sha256: str
    bank_hashes: dict[int, tuple[str, str]]


def load_comparison_config(path: Path) -> ComparisonConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference_index = payload.get("candidate_teacher_reference_index")
    if reference_index is not None and (
        not isinstance(reference_index, int)
        or isinstance(reference_index, bool)
        or reference_index < 0
    ):
        raise ValueError("Candidate reference index is invalid")
    hashes = {
        int(seed): (str(values["train"]), str(values["validation"]))
        for seed, values in payload["bank_hashes"].items()
    }
    config = ComparisonConfig(
        Path(payload["candidate0_path"]),
        CandidateTeacherUsage(payload["candidate_teacher_usage"]),
        reference_index,
        Path(payload["output_dir"]),
        Path(payload["run_dir"]),
        int(payload["data_seed"]),
        tuple(int(value) for value in payload["bank_seeds"]),
        tuple(int(value) for value in payload["model_seeds"]),
        int(payload["trials"]),
        int(payload["steps"]),
        tuple(int(value) for value in payload["checkpoints"]),
        float(payload["learning_rate"]),
        int(payload["batch_size"]),
        float(payload["monitor_interval_seconds"]),
        int(payload["stall_intervals"]),
        str(payload["candidate0_rf_sha256"]),
        hashes,
    )
    if config.bank_seeds != (31001, 31002, 31003):
        raise ValueError("canonical bank seeds must remain 31001/31002/31003")
    if config.model_seeds != (19, 20, 21) or config.trials != 2:
        raise ValueError("canonical model seeds and T=2 are frozen")
    if config.steps != 400 or config.checkpoints != (0, 50, 100, 200, 400):
        raise ValueError("canonical final checkpoint is fixed at step 400")
    return config


__all__ = ["ComparisonConfig", "load_comparison_config"]
