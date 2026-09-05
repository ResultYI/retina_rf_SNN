from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Final

import torch

from evaluation.mechanistic_retina.clean_sampled_artifacts import (
    save_clean_checkpoint,
)
from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig,
    build_clean_state,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    effective_parameter_values,
    rf_bundle,
)
from evaluation.mechanistic_retina.clean_sampled_training import train_clean_student
from evaluation.mechanistic_retina.noise_free_recovery_metrics import (
    comparison,
    counterfactual_comparison,
)
from evaluation.mechanistic_retina.sampled_robustness_metrics import (
    counterfactuals,
    sampled_nll,
    split_data,
    teacher_probability_metrics,
)


_STUDENT_SEEDS: Final = (53_001, 53_002, 53_003)
_SPIKE_SEEDS: Final = (54_001, 54_002, 54_003)
type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _run_key(student_seed: int, spike_seed: int) -> str:
    return f"student-{student_seed}_spike-{spike_seed}"


def _rf_comparison(
    teacher: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: {
            "raw": comparison(teacher[name], raw[name]),
            "trained": comparison(teacher[name], trained[name]),
        }
        for name in teacher
    }


def _parameter_comparison(
    teacher: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: {
            "raw": comparison(teacher[name], raw[name]),
            "trained": comparison(teacher[name], trained[name]),
        }
        for name in teacher
    }


def run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("sampled robustness output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = tuple(
        dict.fromkeys(
            (
                *((seed, _SPIKE_SEEDS[0]) for seed in _STUDENT_SEEDS),
                *((_STUDENT_SEEDS[0], seed) for seed in _SPIKE_SEEDS),
            )
        )
    )
    results: dict[str, JsonValue] = {}
    parameter_tensors: dict[str, dict[str, torch.Tensor]] = {}
    rf_tensors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    counterfactual_tensors: dict[
        str, dict[str, dict[str, dict[str, torch.Tensor]]]
    ] = {}
    teacher_saved = False
    for student_seed, spike_seed in combinations:
        state = build_clean_state(
            CleanBenchmarkConfig(
                student_seed=student_seed,
                spike_seed=spike_seed,
            )
        )
        key = _run_key(student_seed, spike_seed)
        train_split = split_data(state, validation=False)
        validation_split = split_data(state, validation=True)
        context_cones = validation_split.cones[:2]
        context_history = validation_split.spikes[:2]
        teacher_parameters = effective_parameter_values(state.teacher)
        raw_parameters = effective_parameter_values(state.student)
        teacher_rf = rf_bundle(state.teacher, context_cones, context_history)
        raw_rf = rf_bundle(state.student, context_cones, context_history)
        teacher_counterfactual, teacher_effects = counterfactuals(
            state.teacher, validation_split
        )
        raw_sampled = {
            "train": sampled_nll(state.student, train_split),
            "validation": sampled_nll(state.student, validation_split),
        }
        raw_kl = {
            "train": teacher_probability_metrics(
                state.student, state.teacher, train_split
            ),
            "validation": teacher_probability_metrics(
                state.student, state.teacher, validation_split
            ),
        }
        evidence = train_clean_student(state)
        trained_parameters = effective_parameter_values(state.student)
        trained_rf = rf_bundle(state.student, context_cones, context_history)
        student_counterfactual, student_effects = counterfactuals(
            state.student, validation_split
        )
        trained_sampled = {
            "train": sampled_nll(state.student, train_split),
            "validation": sampled_nll(state.student, validation_split),
        }
        trained_kl = {
            "train": teacher_probability_metrics(
                state.student, state.teacher, train_split
            ),
            "validation": teacher_probability_metrics(
                state.student, state.teacher, validation_split
            ),
        }
        results[key] = {
            "student_seed": student_seed,
            "spike_seed": spike_seed,
            "sampled_nll": {"raw": raw_sampled, "trained": trained_sampled},
            "teacher_probability": {"raw": raw_kl, "trained": trained_kl},
            "effective_parameters": _parameter_comparison(
                teacher_parameters, raw_parameters, trained_parameters
            ),
            "rf": _rf_comparison(teacher_rf, raw_rf, trained_rf),
            "counterfactuals": {
                "teacher": teacher_effects,
                "student": student_effects,
                "teacher_student": counterfactual_comparison(
                    teacher_counterfactual, student_counterfactual
                ),
            },
            "training_evidence": {
                "pathway_updates": evidence.pathway_updates,
                "cell_gain_updates": evidence.cell_gain_updates,
                "tau_updates": evidence.tau_updates,
                "explicit_delay_updates": evidence.explicit_delay_updates,
            },
        }
        parameter_tensors[key] = {
            **{f"teacher_{name}": value for name, value in teacher_parameters.items()},
            **{f"trained_{name}": value for name, value in trained_parameters.items()},
        }
        rf_tensors[key] = {
            "teacher": teacher_rf,
            "raw": raw_rf,
            "trained": trained_rf,
        }
        counterfactual_tensors[key] = {
            "teacher": teacher_counterfactual,
            "student": student_counterfactual,
        }
        save_clean_checkpoint(
            output_dir / f"{key}-trained.pt",
            state.student,
            state,
            f"{key}-trained",
        )
        torch.save(
            {
                "train_cones": state.train_cones,
                "validation_cones": state.validation_cones,
                "train_spikes": state.train_spikes,
                "validation_spikes": state.validation_spikes,
            },
            output_dir / f"sampled-data-spike-{spike_seed}.pt",
        )
        if not teacher_saved:
            save_clean_checkpoint(
                output_dir / "teacher.pt", state.teacher, state, "teacher"
            )
            teacher_saved = True
    payload = {
        "protocol": asdict(CleanBenchmarkConfig()),
        "training_target": "sampled_rgc_spikes_only",
        "fixed_spike_sample": [
            _run_key(seed, _SPIKE_SEEDS[0]) for seed in _STUDENT_SEEDS
        ],
        "fixed_student_protocol": [
            _run_key(_STUDENT_SEEDS[0], seed) for seed in _SPIKE_SEEDS
        ],
        "shared_baseline": _run_key(_STUDENT_SEEDS[0], _SPIKE_SEEDS[0]),
        "runs": results,
        "numerical_anomaly_detected": False,
    }
    torch.save(parameter_tensors, output_dir / "effective-parameter-tensors.pt")
    torch.save(rf_tensors, output_dir / "rf-tensors.pt")
    torch.save(counterfactual_tensors, output_dir / "counterfactual-tensors.pt")
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
