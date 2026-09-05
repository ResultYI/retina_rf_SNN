from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Final

import torch

from evaluation.mechanistic_retina.clean_sampled_artifacts import (
    save_clean_checkpoint,
)
from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig,
    build_clean_state,
    nested_budget_state,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    effective_parameter_values,
    rf_bundle,
)
from evaluation.mechanistic_retina.clean_sampled_training import train_clean_student
from evaluation.mechanistic_retina.noise_free_recovery_metrics import comparison
from evaluation.mechanistic_retina.sampled_robustness_metrics import (
    SampledSplit,
    counterfactuals,
    sampled_nll,
    split_data,
    teacher_probability_metrics,
)


_REPEAT_BUDGETS: Final = (1, 2, 4, 8, 16)
_SPIKE_SEEDS: Final = (54_001, 54_002, 54_003)
type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _recovery(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    record = comparison(reference, value)
    teacher_norm = record["teacher_norm"]
    record["relative_l2"] = (
        record["difference_norm"] / teacher_norm if teacher_norm else math.inf
    )
    record["norm_ratio"] = (
        record["student_norm"] / teacher_norm if teacher_norm else math.inf
    )
    return record


def _recovery_bundle(
    teacher: dict[str, torch.Tensor], value: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    return {name: _recovery(teacher[name], value[name]) for name in teacher}


def _counterfactual_recovery(
    teacher: dict[str, dict[str, torch.Tensor]],
    student: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        pathway: {
            signal: _recovery(teacher[pathway][signal], student[pathway][signal])
            for signal in ("logit_delta", "probability_delta", "rf_delta")
        }
        for pathway in teacher
    }


def _observations(split: SampledSplit) -> dict[str, int]:
    sequences, time_steps, cells = split.spikes.shape
    return {
        "sequence_time_bins": sequences * time_steps,
        "bernoulli_cell_bins": split.spikes.numel(),
        "spike_count": int(split.spikes.sum()),
        "cells": cells,
    }


def _run_key(repeat_budget: int, spike_seed: int) -> str:
    return f"repeats-{repeat_budget}_spike-{spike_seed}"


def run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("nested spike-budget output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, JsonValue] = {}
    tensors: dict[str, JsonValue] = {}
    teacher_saved = False
    for spike_seed in _SPIKE_SEEDS:
        master = build_clean_state(
            CleanBenchmarkConfig(trials=max(_REPEAT_BUDGETS), spike_seed=spike_seed)
        )
        torch.save(
            {
                "train_cones": master.train_cones,
                "validation_cones": master.validation_cones,
                "train_spikes": master.train_spikes,
                "validation_spikes": master.validation_spikes,
            },
            output_dir / f"master-bank-spike-{spike_seed}.pt",
        )
        if not teacher_saved:
            save_clean_checkpoint(
                output_dir / "teacher.pt", master.teacher, master, "teacher"
            )
            teacher_saved = True
        for repeat_budget in _REPEAT_BUDGETS:
            state = nested_budget_state(master, repeat_budget)
            key = _run_key(repeat_budget, spike_seed)
            train_split = split_data(state, validation=False)
            validation_split = split_data(state, validation=True)
            fixed_probe = SampledSplit(
                state.validation_cones,
                state.validation_spikes[:, 0],
            )
            context_cones = fixed_probe.cones[:2]
            context_history = fixed_probe.spikes[:2]
            teacher_parameters = effective_parameter_values(state.teacher)
            raw_parameters = effective_parameter_values(state.student)
            teacher_rf = rf_bundle(state.teacher, context_cones, context_history)
            raw_rf = rf_bundle(state.student, context_cones, context_history)
            teacher_counterfactual, _ = counterfactuals(
                state.teacher, fixed_probe
            )
            raw_metrics = {
                "sampled_nll": {
                    "train": sampled_nll(state.student, train_split),
                    "validation": sampled_nll(state.student, validation_split),
                },
                "teacher_probability": {
                    "train": teacher_probability_metrics(
                        state.student, state.teacher, train_split
                    ),
                    "validation": teacher_probability_metrics(
                        state.student, state.teacher, validation_split
                    ),
                },
            }
            evidence = train_clean_student(state)
            trained_parameters = effective_parameter_values(state.student)
            trained_rf = rf_bundle(state.student, context_cones, context_history)
            student_counterfactual, _ = counterfactuals(
                state.student, fixed_probe
            )
            trained_metrics = {
                "sampled_nll": {
                    "train": sampled_nll(state.student, train_split),
                    "validation": sampled_nll(state.student, validation_split),
                },
                "teacher_probability": {
                    "train": teacher_probability_metrics(
                        state.student, state.teacher, train_split
                    ),
                    "validation": teacher_probability_metrics(
                        state.student, state.teacher, validation_split
                    ),
                },
            }
            results[key] = {
                "repeat_budget": repeat_budget,
                "spike_seed": spike_seed,
                "observations": {
                    "train": _observations(train_split),
                    "validation": _observations(validation_split),
                },
                "raw": raw_metrics,
                "trained": trained_metrics,
                "effective_parameters": {
                    "raw": _recovery_bundle(teacher_parameters, raw_parameters),
                    "trained": _recovery_bundle(
                        teacher_parameters, trained_parameters
                    ),
                },
                "rf": {
                    "raw": _recovery_bundle(teacher_rf, raw_rf),
                    "trained": _recovery_bundle(teacher_rf, trained_rf),
                },
                "counterfactuals": _counterfactual_recovery(
                    teacher_counterfactual, student_counterfactual
                ),
                "training_updates": {
                    "pathways": evidence.pathway_updates,
                    "cell_gains": evidence.cell_gain_updates,
                    "tau": evidence.tau_updates,
                    "explicit_delay": evidence.explicit_delay_updates,
                },
            }
            tensors[key] = {
                "effective_parameters": {
                    "teacher": teacher_parameters,
                    "raw": raw_parameters,
                    "trained": trained_parameters,
                },
                "rf": {"teacher": teacher_rf, "raw": raw_rf, "trained": trained_rf},
                "counterfactuals": {
                    "teacher": teacher_counterfactual,
                    "student": student_counterfactual,
                },
            }
            save_clean_checkpoint(
                output_dir / f"{key}-trained.pt", state.student, state, key
            )
    payload = {
        "protocol": asdict(CleanBenchmarkConfig()),
        "repeat_budgets": list(_REPEAT_BUDGETS),
        "spike_seeds": list(_SPIKE_SEEDS),
        "nesting": "each budget is the leading repeat prefix of one 16-repeat master bank per spike seed",
        "training_target": "sampled_rgc_spikes_only",
        "rf_probe": "first repeat of the first two validation stimuli",
        "counterfactual_probe": "first repeat of every validation stimulus",
        "runs": results,
    }
    torch.save(tensors, output_dir / "recovery-tensors.pt")
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
