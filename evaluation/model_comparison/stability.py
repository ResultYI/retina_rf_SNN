from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence

import torch

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.rf import rf_cosine
from evaluation.model_comparison.types import RunResult


def stability_payload(runs: Sequence[RunResult]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for model in _model_order(runs):
        selected = tuple(run for run in runs if run.model == model)
        ce_values = tuple(run.prediction.teacher_expected_ce for run in selected)
        rf_runs = tuple(run for run in selected if run.rf_tensor is not None)
        entry: dict[str, JsonValue] = {
            "run_count": len(selected),
            "teacher_ce_range": [min(ce_values), max(ce_values)],
            "prediction_variance": _prediction_variance(selected),
            "bank_level_ce": _means_by_bank(selected),
            "seed_level_ce": _means_by_seed(selected),
        }
        if rf_runs:
            entry.update(
                {
                    "all_run_rf_cosine": _pairwise_rf(rf_runs),
                    "cross_seed_rf_cosine": _cross_seed(rf_runs),
                    "cross_bank_rf_cosine": _cross_bank(rf_runs),
                    "per_cell_rf_variance": _per_cell_variance(rf_runs),
                    "exact_cell_stability": _exact_stability(rf_runs),
                }
            )
        auxiliary = tuple(run for run in selected if run.auxiliary_tensor is not None)
        if auxiliary:
            entry["subunit_stability"] = _pairwise_auxiliary(auxiliary)
        payload[model] = entry
    payload["paired_comparisons"] = {
        "mechanistic_vs_glm": _paired_win_rates(runs, "GLM-SH"),
        "mechanistic_vs_graph_tcn": _paired_win_rates(runs, "Graph-TCN"),
        "mechanistic_vs_lnln": _paired_win_rates(runs, "LN-LN"),
    }
    return payload


def _model_order(runs: Sequence[RunResult]) -> tuple[str, ...]:
    canonical = ("Bias", "GLM-SH", "LN-LN", "Graph-TCN", "Mechanistic Retina")
    return tuple(model for model in canonical if any(run.model == model for run in runs))


def _means_by_bank(runs: Sequence[RunResult]) -> dict[str, JsonValue]:
    return {
        str(bank): statistics.fmean(
            run.prediction.teacher_expected_ce for run in runs if run.bank_seed == bank
        )
        for bank in sorted({run.bank_seed for run in runs})
    }


def _means_by_seed(runs: Sequence[RunResult]) -> dict[str, JsonValue]:
    seeds = sorted({run.model_seed for run in runs if run.model_seed is not None})
    return {
        str(seed): statistics.fmean(
            run.prediction.teacher_expected_ce for run in runs if run.model_seed == seed
        )
        for seed in seeds
    }


def _prediction_variance(runs: Sequence[RunResult]) -> float:
    if len(runs) < 2:
        return 0.0
    return float(torch.stack(tuple(run.mean_logits for run in runs)).var(dim=0, unbiased=False).mean())


def _pairwise_rf(runs: Sequence[RunResult]) -> float:
    pairs = tuple(itertools.combinations(runs, 2))
    return statistics.fmean(
        rf_cosine(_rf(left), _rf(right)) for left, right in pairs
    ) if pairs else 1.0


def _pairwise_auxiliary(runs: Sequence[RunResult]) -> float:
    pairs = tuple(itertools.combinations(runs, 2))
    return statistics.fmean(
        rf_cosine(_auxiliary(left), _auxiliary(right)) for left, right in pairs
    ) if pairs else 1.0


def _cross_seed(runs: Sequence[RunResult]) -> float | None:
    pairs = tuple(
        (left, right)
        for left, right in itertools.combinations(runs, 2)
        if left.bank_seed == right.bank_seed and left.model_seed != right.model_seed
    )
    return statistics.fmean(rf_cosine(_rf(a), _rf(b)) for a, b in pairs) if pairs else None


def _cross_bank(runs: Sequence[RunResult]) -> float | None:
    pairs = tuple(
        (left, right)
        for left, right in itertools.combinations(runs, 2)
        if left.bank_seed != right.bank_seed and left.model_seed == right.model_seed
    )
    if not pairs and all(run.model_seed is None for run in runs):
        pairs = tuple(itertools.combinations(runs, 2))
    return statistics.fmean(rf_cosine(_rf(a), _rf(b)) for a, b in pairs) if pairs else None


def _per_cell_variance(runs: Sequence[RunResult]) -> list[JsonValue]:
    tensors = torch.stack(tuple(_rf(run).mean(dim=0) for run in runs))
    values = tensors.var(dim=0, unbiased=False).flatten(1).mean(dim=1)
    return [float(value) for value in values]


def _exact_stability(runs: Sequence[RunResult]) -> float:
    resolved = torch.tensor(
        tuple(
            tuple(cell.exact_resolved for cell in run.rf.summary.metric.cells)
            for run in runs
            if run.rf is not None
        )
    )
    return float(resolved.all(dim=0).float().mean())


def _paired_win_rates(runs: Sequence[RunResult], baseline: str) -> dict[str, JsonValue]:
    main = tuple(run for run in runs if run.model == "Mechanistic Retina")
    comparisons = []
    for run in main:
        candidates = tuple(
            other
            for other in runs
            if other.model == baseline
            and other.bank_seed == run.bank_seed
            and (other.model_seed in {None, run.model_seed})
        )
        if candidates:
            comparisons.append((run, candidates[0]))
    return {
        "pair_count": len(comparisons),
        "ce_win_rate": sum(a.prediction.teacher_expected_ce < b.prediction.teacher_expected_ce for a, b in comparisons) / max(len(comparisons), 1),
        "rf_win_rate": sum(a.rf is not None and b.rf is not None and a.rf.summary.metric.global_cosine > b.rf.summary.metric.global_cosine for a, b in comparisons) / max(len(comparisons), 1),
    }


def _rf(run: RunResult) -> torch.Tensor:
    if run.rf_tensor is None:
        raise ValueError("RF stability requires an RF tensor")
    return run.rf_tensor


def _auxiliary(run: RunResult) -> torch.Tensor:
    if run.auxiliary_tensor is None:
        raise ValueError("auxiliary stability requires a tensor")
    return run.auxiliary_tensor


__all__ = ["stability_payload"]
