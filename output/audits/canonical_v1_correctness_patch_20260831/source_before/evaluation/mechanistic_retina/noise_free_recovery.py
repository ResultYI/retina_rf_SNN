from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

from evaluation.mechanistic_retina.clean_sampled_artifacts import (
    save_clean_checkpoint,
)
from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig,
    CleanBenchmarkState,
    build_clean_state,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    effective_parameter_values,
    rf_bundle,
)
from evaluation.mechanistic_retina.noise_free_recovery_metrics import (
    comparison,
    counterfactual_comparison,
    counterfactuals,
    expected_metrics,
    probability,
    temporal_values,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer


_STUDENT_SEEDS = (53_001, 53_002, 53_003)
type JsonScalar = None | bool | int | float | str
type JsonValue = (
    JsonScalar | list["JsonValue"] | tuple["JsonValue", ...] | dict[str, "JsonValue"]
)


def _fit(
    model: MechanisticGraphTemporalRetina,
    state: CleanBenchmarkState,
    train_target: torch.Tensor,
) -> None:
    config = state.config
    optimizer = build_phase1_optimizer(model, learning_rate=config.learning_rate)
    generator = torch.Generator().manual_seed(config.training_seed)
    for _ in range(config.steps):
        indices = torch.randint(
            state.train_cones.shape[0],
            (config.batch_size,),
            generator=generator,
        )
        target = train_target[indices]
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_sequence(
            state.train_cones[indices], observed_counts=torch.zeros_like(target)
        ).logits
        loss = expected_bernoulli_nll(logits, target, torch.ones_like(target))
        loss.backward()
        optimizer.step()
        model.project_mechanism_parameters()


def _build_student(state: CleanBenchmarkState, seed: int) -> MechanisticGraphTemporalRetina:
    torch.manual_seed(seed)
    return build_mechanistic_retina(
        state.teacher.config,
        state.cone_positions,
        state.cell_positions,
        state.cell_types,
        state.polarities,
    )


def run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("noise-free recovery output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_clean_state(CleanBenchmarkConfig(student_seed=_STUDENT_SEEDS[0]))
    train_target = probability(state.teacher, state.train_cones)
    validation_target = probability(state.teacher, state.validation_cones)
    probe_cones = state.validation_cones[:2]
    probe_history = validation_target.new_zeros(
        (2, state.config.time_steps, validation_target.shape[-1])
    )
    teacher_parameters = effective_parameter_values(state.teacher)
    teacher_rf = rf_bundle(state.teacher, probe_cones, probe_history)
    teacher_counterfactual, teacher_effects = counterfactuals(
        state.teacher, state.validation_cones, teacher_rf["global"]
    )
    save_clean_checkpoint(output_dir / "teacher.pt", state.teacher, state, "teacher")
    torch.save(
        {"train": train_target, "validation": validation_target},
        output_dir / "teacher-probabilities.pt",
    )
    solutions: dict[str, JsonValue] = {}
    parameter_tensors: dict[str, dict[str, torch.Tensor]] = {
        "teacher": teacher_parameters
    }
    rf_tensors: dict[str, dict[str, torch.Tensor]] = {"teacher": teacher_rf}
    perturbation_tensors: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        "teacher": teacher_counterfactual
    }
    validation_kls = []
    for seed in _STUDENT_SEEDS:
        student = _build_student(state, seed)
        raw_parameters = effective_parameter_values(student)
        raw_rf = rf_bundle(student, probe_cones, probe_history)
        raw_metrics = expected_metrics(student, state.validation_cones, validation_target)
        _fit(student, state, train_target)
        trained_parameters = effective_parameter_values(student)
        trained_rf = rf_bundle(student, probe_cones, probe_history)
        trained_metrics = expected_metrics(
            student, state.validation_cones, validation_target
        )
        student_counterfactual, student_effects = counterfactuals(
            student, state.validation_cones, trained_rf["global"]
        )
        key = str(seed)
        validation_kls.append(trained_metrics["kl"])
        solutions[key] = {
            "validation": {"raw": raw_metrics, "trained": trained_metrics},
            "effective_parameters": {
                name: {
                    "raw": comparison(teacher_parameters[name], raw_parameters[name]),
                    "trained": comparison(
                        teacher_parameters[name], trained_parameters[name]
                    ),
                }
                for name in teacher_parameters
            },
            "temporal_parameters": {
                "teacher": temporal_values(state.teacher),
                "trained": temporal_values(student),
            },
            "rf": {
                name: {
                    "raw": comparison(teacher_rf[name], raw_rf[name]),
                    "trained": comparison(teacher_rf[name], trained_rf[name]),
                }
                for name in teacher_rf
            },
            "counterfactual_effects": {
                "teacher": teacher_effects,
                "student": student_effects,
                "teacher_student": counterfactual_comparison(
                    teacher_counterfactual, student_counterfactual
                ),
            },
        }
        parameter_tensors[key] = trained_parameters
        rf_tensors[key] = trained_rf
        perturbation_tensors[key] = student_counterfactual
        save_clean_checkpoint(
            output_dir / f"student-seed-{seed}-trained.pt",
            student,
            state,
            f"student-seed-{seed}-trained",
        )
    payload = {
        "protocol": asdict(state.config),
        "training_target": "teacher_probability_expected_bernoulli_ce",
        "sampled_spikes_used_in_training": False,
        "history": "all-zero deterministic history",
        "fresh_teacher": True,
        "fresh_students": True,
        "student_seeds": list(_STUDENT_SEEDS),
        "teacher_ac_group_mixture": {
            "group_order": [
                "midget_ON",
                "midget_OFF",
                "parasol_ON",
                "parasol_OFF",
            ],
            "local": state.teacher.gates.values(frozenset()).ac_local[:4].tolist(),
            "transient": state.teacher.gates.values(
                frozenset()
            ).ac_transient[:4].tolist(),
        },
        "validation_kl": {
            "mean": statistics.mean(validation_kls),
            "standard_deviation": statistics.pstdev(validation_kls),
            "minimum": min(validation_kls),
            "maximum": max(validation_kls),
        },
        "solutions": solutions,
        "numerical_anomaly_detected": not all(
            torch.isfinite(torch.tensor(value)) for value in validation_kls
        ),
    }
    torch.save(parameter_tensors, output_dir / "trained-parameter-tensors.pt")
    torch.save(rf_tensors, output_dir / "trained-rf-tensors.pt")
    torch.save(perturbation_tensors, output_dir / "perturbation-tensors.pt")
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
