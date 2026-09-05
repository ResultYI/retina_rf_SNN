from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from evaluation.mechanistic_retina.spike_banks import tensor_sha256
from evaluation.model_comparison.prediction import PredictionMetrics
from evaluation.model_comparison.sample_efficiency_reporting import MetricRow
from evaluation.model_comparison.types import RunResult, TrainingPoint

SpyCall = tuple[str, int, int | None, int, str, str]
BankRunner = Callable[..., RunResult]
SeedRunner = Callable[..., RunResult]


def config(tmp_path: Path) -> Path:
    path = tmp_path / "sample-efficiency.json"
    path.write_text(
        json.dumps(
            {
                "canonical_config_path": "configs/model_comparison_t2.yaml",
                "output_dir": str(tmp_path / "evidence"),
                "run_dir": str(tmp_path / "runs"),
                "selection_seed": 19,
                "fractions": [0.25, 0.5, 1.0],
            }
        ),
        encoding="utf-8",
    )
    return path


def counts_by_fraction(rows: Sequence[dict[str, str]]) -> dict[str, int]:  # noqa: DICT_OK
    return {fraction: sum(row["fraction"] == fraction for row in rows) for fraction in {"0.25", "0.5", "1.0"}}


def read_rows(path: Path) -> tuple[dict[str, str], ...]:  # noqa: DICT_OK
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:  # noqa: DICT_OK
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spy_bank_runner(calls: list[SpyCall], model: str, seed: int | None, params: int) -> BankRunner:
    def run(request) -> RunResult:
        calls.append(
            (
                model,
                request.match_target_parameters,
                seed,
                int(request.data.train_cones.shape[0]),
                tensor_sha256(request.data.train_cones),
                tensor_sha256(request.data.validation_cones),
            )
        )
        return fake_run(model, request.bank_seed, seed, params)

    return run


def spy_seed_runner(calls: list[SpyCall], model: str, params: int) -> SeedRunner:
    def run(request, seed: int) -> RunResult:
        return spy_bank_runner(calls, model, seed, params)(request)

    return run


def fake_run(model: str, bank_seed: int, seed: int | None, params: int) -> RunResult:
    prediction = PredictionMetrics(
        0.3,
        0.31,
        0.1,
        0.01,
        0.02,
        tuple(0.3 for _ in range(16)),
        tuple(0.31 for _ in range(16)),
        tuple(0.01 for _ in range(16)),
        tuple(0.02 for _ in range(16)),
    )
    return RunResult(
        model,
        bank_seed,
        seed,
        params,
        prediction,
        None,
        None,
        None,
        torch.zeros(1),
        (TrainingPoint(0, 0.3, 0.0),),
        True,
        {},
    )


def call_counts(calls: Sequence[SpyCall]) -> dict[tuple[str, int], int]:
    return {
        (model, target): sum(call[0] == model and call[1] == target for call in calls)
        for model, target in {(call[0], call[1]) for call in calls}
    }


def wait_for_lock(path: Path) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError("lock file was not acquired")


def failing_provider(fraction: float, train_count: int) -> tuple[MetricRow, ...]:
    raise AssertionError(f"provider called for {fraction} {train_count}")
