from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig,
    CleanBenchmarkState,
    build_clean_state,
)
from evaluation.mechanistic_retina.clean_sampled_artifacts import (
    save_clean_checkpoint,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    JsonValue,
    bounded_parameter_learning_confirmed,
    effective_parameter_recovery_summary,
    effective_parameter_values,
    explicit_delay_order_valid,
    rf_bundle,
    rf_summary,
    tensor_change_record,
)
from evaluation.mechanistic_retina.clean_sampled_training import train_clean_student
from evaluation.mechanistic_retina.spike_banks import tensor_sha256
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from models.mechanistic_retina.contracts import PathwayClamp
from training.mechanistic_retina.losses import expected_bernoulli_nll


@dataclass(frozen=True, slots=True)
class CleanBenchmarkResult:
    training_target: str
    loaded_checkpoint: bool
    tau_parameters_learned: bool
    explicit_pathway_delays_learned: bool
    validation_nll_raw: float
    validation_nll_trained: float
    artifact_dir: Path


def build_clean_benchmark(config: CleanBenchmarkConfig) -> CleanBenchmarkState:
    return build_clean_state(config)


def run_clean_benchmark(
    config: CleanBenchmarkConfig,
    output_dir: Path,
    *,
    pathway_diagnostics: bool = False,
) -> CleanBenchmarkResult:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("clean benchmark output directory must be empty")
    state = build_clean_state(
        config,
        verify_teacher_pathways=pathway_diagnostics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_clean_checkpoint(output_dir / "teacher.pt", state.teacher, state, "teacher")
    save_clean_checkpoint(
        output_dir / "student-raw.pt", state.student, state, "student-raw"
    )
    torch.save(
        {
            "train_cones": state.train_cones,
            "validation_cones": state.validation_cones,
            "train_spikes": state.train_spikes,
            "validation_spikes": state.validation_spikes,
        },
        output_dir / "sampled-data.pt",
    )
    context_cones = state.validation_cones[:2]
    context_spikes = state.validation_spikes[:2, 0]
    raw_rf = rf_bundle(
        state.student,
        context_cones,
        context_spikes,
        include_pathways=pathway_diagnostics,
    )
    teacher_rf = rf_bundle(
        state.teacher,
        context_cones,
        context_spikes,
        include_pathways=pathway_diagnostics,
    )
    teacher_effective = effective_parameter_values(state.teacher)
    raw_effective = effective_parameter_values(state.student)
    raw_predictions = _prediction_tensors(state.student, state)
    evidence = train_clean_student(state)
    trained_predictions = _prediction_tensors(state.student, state)
    trained_effective = effective_parameter_values(state.student)
    trained_rf = rf_bundle(
        state.student,
        context_cones,
        context_spikes,
        include_pathways=pathway_diagnostics,
    )
    save_clean_checkpoint(
        output_dir / "student-trained.pt", state.student, state, "student-trained"
    )
    torch.save(
        {"teacher": teacher_rf, "raw": raw_rf, "trained": trained_rf},
        output_dir / "rf-tensors.pt",
    )
    torch.save(
        {"raw": raw_predictions, "trained": trained_predictions},
        output_dir / "prediction-tensors.pt",
    )
    recovery_tensors = {
        "teacher": teacher_effective,
        "raw": raw_effective,
        "trained": trained_effective,
    }
    torch.save(recovery_tensors, output_dir / "effective-parameter-recovery.pt")
    structural_tensors, structural_summary = _structural_ablation(state)
    torch.save(structural_tensors, output_dir / "structural-ablation.pt")
    summary = rf_summary(teacher_rf, raw_rf, trained_rf)
    tau_learned = bounded_parameter_learning_confirmed(evidence.tau_updates)
    delays_learned = bounded_parameter_learning_confirmed(
        evidence.explicit_delay_updates
    )
    delay_order_is_valid = explicit_delay_order_valid(state.student)
    delays_are_finite = all(
        record["finite"] is True for record in evidence.explicit_delay_updates.values()
    )
    delay_hit_boundary = any(
        record["hit_boundary"] is True
        for record in evidence.explicit_delay_updates.values()
    )
    payload: dict[str, JsonValue] = {
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "causal_contract": state.student.config.causal_contract,
        "bc_encoder_parameters": "one shared normalized BC encoder for direct and broad support views",
        "training_target": "sampled_rgc_spikes_only",
        "loaded_checkpoint": False,
        "fresh_optimizer": True,
        "pathway_diagnostics": pathway_diagnostics,
        "tau_parameters_learned": tau_learned,
        "explicit_pathway_delays_learned": delays_learned,
        "timing_concepts": {
            "tau": "bounded-learnable filter/state decay time constant in ms",
            "explicit_pathway_delay": "bounded-learnable causal fractional H1/shared-BC/AC-downstream delay in ms",
            "rf_lag_window": "fixed non-learnable truncated RF analysis window",
            "rgc_history_shift": "fixed non-learnable one-bin causal observed-spike shift",
        },
        "rf_lag_window": {
            "learnable": False,
            "lag_steps": state.student.config.lag_steps,
            "dt_ms": state.student.config.dt_ms,
            "maximum_lag_ms": (state.student.config.lag_steps - 1)
            * state.student.config.dt_ms,
        },
        "rgc_history_shift": {
            "learnable": False,
            "shift_steps": 1,
            "shift_ms": state.student.config.dt_ms,
        },
        "config": asdict(config),
        "data_sha256": {
            "train_cones": tensor_sha256(state.train_cones),
            "validation_cones": tensor_sha256(state.validation_cones),
            "train_spikes": tensor_sha256(state.train_spikes),
            "validation_spikes": tensor_sha256(state.validation_spikes),
        },
        "validation_spike_nll": {
            "raw": evidence.validation_nll[0]["nll"],
            "trained": evidence.validation_nll[-1]["nll"],
            "trajectory": evidence.validation_nll,
        },
        "rf": summary,
        "temporal_rf_change": tensor_change_record(
            raw_rf["temporal"], trained_rf["temporal"]
        ),
        "pathway_updates": evidence.pathway_updates,
        "cell_gain_updates": evidence.cell_gain_updates,
        "effective_parameter_recovery": effective_parameter_recovery_summary(
            teacher_effective, raw_effective, trained_effective
        ),
        "effective_parameter_values": {
            phase: {name: value.tolist() for name, value in values.items()}
            for phase, values in recovery_tensors.items()
        },
        "structural_ablation": structural_summary,
        "seeds": {
            "stimulus": config.stimulus_seed,
            "teacher": config.teacher_seed,
            "student": config.student_seed,
            "spike": config.spike_seed,
            "training": config.training_seed,
        },
        "tau_parameters": evidence.tau_updates,
        "explicit_pathway_delay_parameters": evidence.explicit_delay_updates,
        "explicit_pathway_delay_diagnostics": {
            "all_finite": delays_are_finite,
            "order_valid": delay_order_is_valid,
            "any_hit_boundary": delay_hit_boundary,
        },
        "optimizer_parameters": evidence.optimizer_parameters,
        "parameter_inventory": evidence.parameter_inventory,
        "numerical_anomaly_detected": False,
    }
    if pathway_diagnostics:
        payload["teacher_pathway_effects"] = state.teacher_effects
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return CleanBenchmarkResult(
        "sampled_rgc_spikes_only",
        False,
        tau_learned,
        delays_learned,
        float(evidence.validation_nll[0]["nll"]),
        float(evidence.validation_nll[-1]["nll"]),
        output_dir,
    )


def _prediction_tensors(
    model: torch.nn.Module,
    state: CleanBenchmarkState,
    *,
    clamps: frozenset[PathwayClamp] = frozenset(),
) -> dict[str, torch.Tensor]:
    stimuli, trials, time_steps, cells = state.validation_spikes.shape
    cones = state.validation_cones[:, None].expand(-1, trials, -1, -1).reshape(
        stimuli * trials, time_steps, -1
    )
    spikes = state.validation_spikes.reshape(stimuli * trials, time_steps, cells)
    with torch.no_grad():
        output = model.forward_sequence(cones, observed_counts=spikes, clamps=clamps)
    return {
        "logits": output.logits.reshape(stimuli, trials, time_steps, cells),
        "spike_probability": output.spike_probability.reshape(
            stimuli, trials, time_steps, cells
        ),
        "target_spikes": state.validation_spikes,
        "bc_current": (output.bc_sustained_current + output.bc_transient_current).reshape(
            stimuli, trials, time_steps, cells
        ),
        "bc_broad_presynaptic": output.bc_broad_presynaptic.reshape(
            stimuli, trials, time_steps, cells, 2
        ),
        "ac_current": (
            output.amacrine_local_current + output.amacrine_transient_current
        ).reshape(stimuli, trials, time_steps, cells),
        "h1_current": output.h1_surround_contribution.reshape(
            stimuli, trials, time_steps, -1
        ),
    }


def _structural_ablation(
    state: CleanBenchmarkState,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, JsonValue]]]:
    normal = _prediction_tensors(state.student, state)
    clamps = {
        "H1_off": frozenset({PathwayClamp.H1}),
        "direct_BC_off": frozenset(
            {PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}
        ),
        "AC_off": frozenset(
            {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
        ),
    }
    tensors: dict[str, dict[str, torch.Tensor]] = {"normal": normal}
    summary: dict[str, dict[str, JsonValue]] = {}
    for name, pathway_clamps in clamps.items():
        clamped = _prediction_tensors(state.student, state, clamps=pathway_clamps)
        tensors[name] = clamped
        current_name = {
            "H1_off": "h1_current",
            "direct_BC_off": "bc_current",
            "AC_off": "ac_current",
        }[name]
        assert torch.count_nonzero(clamped[current_name]) == 0
        if PathwayClamp.DIRECT_BC_SUSTAINED in pathway_clamps:
            assert torch.equal(clamped["bc_broad_presynaptic"], normal["bc_broad_presynaptic"])
            assert torch.equal(clamped["ac_current"], normal["ac_current"])
        summary[name] = {
            "validation_nll": float(
                expected_bernoulli_nll(
                    clamped["logits"],
                    clamped["target_spikes"],
                    torch.ones_like(clamped["target_spikes"]),
                )
            ),
            "mean_absolute_logit_change": float(
                (clamped["logits"] - normal["logits"]).abs().mean()
            ),
            "mean_absolute_probability_change": float(
                (
                    clamped["spike_probability"] - normal["spike_probability"]
                ).abs().mean()
            ),
            "structural_current_exact_zero": bool(
                torch.count_nonzero(clamped[current_name]) == 0
            ),
            "structural_current_max_abs": float(clamped[current_name].abs().max()),
        }
    return tensors, summary


__all__ = [
    "CleanBenchmarkConfig",
    "CleanBenchmarkResult",
    "build_clean_benchmark",
    "run_clean_benchmark",
]
