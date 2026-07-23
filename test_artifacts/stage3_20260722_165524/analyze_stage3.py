from __future__ import annotations

import json
import math
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
RUN_DIR = ARTIFACT_DIR / "run"
RUNTIME_SECONDS = 3534.151


def main() -> int:
    rows = [
        json.loads(line)
        for line in (RUN_DIR / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = json.loads((RUN_DIR / "evaluation.json").read_text(encoding="utf-8"))
    environment = json.loads((ARTIFACT_DIR / "environment.json").read_text(encoding="utf-8"))
    audit = json.loads((ARTIFACT_DIR / "gradient_audit.json").read_text(encoding="utf-8"))
    resume_text = (ARTIFACT_DIR / "resume_check.log").read_text(encoding="utf-16")
    trace_keys = (
        "optimizer_step", "hard_energy", "surrogate_budget_energy",
        "reference_energy", "current_budget", "target_budget",
        "current_energy_ratio", "energy_violation", "energy_penalty",
        "validation_mse", "best_reconstruction_event", "best_feasible_event",
        "lr_model", "lr_decoder",
    )
    trace = []
    for row in rows:
        item = {key: row[key] for key in trace_keys}
        item["target_energy_ratio"] = row["validation_target_energy_ratio"]
        item["dual"] = row["energy_dual"]
        item["representation_skill"] = row["validation_representation_skill"]
        trace.append(item)
    (ARTIFACT_DIR / "budget_trace.json").write_text(
        json.dumps(trace, indent=2),
        encoding="utf-8",
    )

    bootstrap_rows = [row for row in rows if row["optimizer_step"] <= 40]
    ramp_rows = [row for row in rows if 40 < row["optimizer_step"] <= 100]
    post_ramp_rows = [row for row in rows if row["optimizer_step"] >= 100]
    reference = rows[1]["reference_energy"]
    target = rows[2]["target_budget"]
    target_ratio = target / reference
    bootstrap_correct = all(
        row["current_budget"] is None
        and row["target_budget"] is None
        and row["energy_violation"] == 0.0
        and row["energy_penalty"] == 0.0
        and row["energy_dual"] == 0.0
        for row in bootstrap_rows
    )
    reference_frozen = all(
        math.isclose(row["reference_energy"], reference, rel_tol=0.0, abs_tol=1e-12)
        for row in rows[1:]
    )
    target_frozen = all(
        math.isclose(row["target_budget"], target, rel_tol=0.0, abs_tol=1e-12)
        for row in rows[2:]
    )
    ramp_errors = []
    for row in ramp_rows:
        fraction = min(1.0, (row["optimizer_step"] - 40) / 60)
        expected = reference + fraction * (target - reference)
        ramp_errors.append(abs(row["current_budget"] - expected) / target)
    ramp_correct = max(ramp_errors) <= 1e-6 and all(
        math.isclose(row["current_budget"], target, rel_tol=1e-6, abs_tol=0.0)
        for row in post_ramp_rows
    )
    hard_surrogate_difference = max(
        abs(row["hard_energy"] - row["surrogate_budget_energy"]) for row in rows
    )
    finite = all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    dual_correct = all(
        row["energy_violation"] >= 0.0
        and row["energy_penalty"] >= 0.0
        and 0.0 <= row["energy_dual"] <= 10.0
        for row in rows
    ) and rows[-1]["energy_dual"] > 0.0
    aggregate_violation_rows = [
        row["optimizer_step"]
        for row in rows
        if row["current_budget"] is not None
        and row["surrogate_budget_energy"] <= row["current_budget"]
        and row["energy_violation"] > 0.0
    ]
    activity_correct = (
        any(row["gradient_norm"] > 0.0 for row in rows)
        and any(row["temporal_gradient_norm"] > 0.0 for row in rows)
        and any(row["mean_rate"] > 0.0 for row in rows)
        and any(
            row["hard_active_fraction_on"] > 0.0
            or row["hard_active_fraction_off"] > 0.0
            for row in post_ramp_rows
        )
        and rows[-1]["validation_mse"] <= 2.0 * rows[0]["validation_mse"]
    )
    feasible_expected = any(
        row["validation_target_energy_ratio"] <= 1.05 for row in post_ramp_rows
    )
    checkpoint_state = {
        "checkpoint_last_exists": (RUN_DIR / "checkpoint_last.pt").exists(),
        "checkpoint_best_reconstruction_exists": (RUN_DIR / "checkpoint_best_reconstruction.pt").exists(),
        "checkpoint_best_feasible_exists": (RUN_DIR / "checkpoint_best_feasible.pt").exists(),
        "initial_reference_exists": (RUN_DIR / "initial_reference.pt").exists(),
    }
    best_feasible_correct = (
        checkpoint_state["checkpoint_best_feasible_exists"] == feasible_expected
        and not any(row["best_feasible_event"] for row in bootstrap_rows)
    )
    resume_tokens = (
        "exit_code=0", "training_rows_before=8", "training_rows_after=8",
        "revision=3", "optimizer_step=160", "'augmentation'", "'sampling'",
    )
    resume_passed = all(token in resume_text for token in resume_tokens)
    expected_lr_model = 0.0002 * 0.5 * (
        1.0 + math.cos(math.pi * rows[-1]["optimizer_step"] / 6000)
    )
    scheduler_correct = (
        evaluation["resolved_config"]["training"]["max_optimizer_steps"] == 6000
        and math.isclose(rows[-1]["lr_model"], expected_lr_model, rel_tol=1e-12)
        and rows[-1]["lr_model"] > 0.0
    )
    audit_mechanical_pass = audit["finite"] and audit["reconstruction_grad_norm"] > 0.0
    required_checkpoints = all(
        checkpoint_state[key]
        for key in (
            "checkpoint_last_exists",
            "checkpoint_best_reconstruction_exists",
            "initial_reference_exists",
        )
    )
    scientific_not_run = (
        evaluation["summary"]["dynamic_rf_status"] == "not_run"
        and evaluation["summary"]["rgc_type_status"] == "not_run"
    )
    stage_passed = all((
        len(rows) == 8, rows[-1]["optimizer_step"] == 160, finite,
        bootstrap_correct, reference_frozen, target_frozen,
        abs(target_ratio - 0.9) <= 1e-6, ramp_correct,
        hard_surrogate_difference <= 1e-7, dual_correct, activity_correct,
        best_feasible_correct, resume_passed, scheduler_correct,
        audit_mechanical_pass, required_checkpoints, scientific_not_run,
    ))
    auxiliary_pressure_acceptable = (
        audit["wiring_to_reconstruction_ratio"] <= 0.25
        and audit["diversity_to_reconstruction_ratio"] <= 0.25
    )
    status = "STAGE_3_PASS" if stage_passed else "STAGE_3_FAIL"
    stage_4_pilot = "GO" if stage_passed and auxiliary_pressure_acceptable else "NO-GO"
    metrics = {
        "optimizer_steps": rows[-1]["optimizer_step"],
        "validation_count": len(rows),
        "runtime_seconds": RUNTIME_SECONDS,
        "scheduler_horizon_steps": 6000,
        "final_lr_model": rows[-1]["lr_model"],
        "expected_final_lr_model": expected_lr_model,
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in rows),
        "peak_memory_ratio": max(row["peak_memory_bytes"] for row in rows)
        / environment["gpu_total_memory_bytes"],
        "reference_energy": reference,
        "target_budget": target,
        "target_to_reference_ratio": target_ratio,
        "bootstrap_correct": bootstrap_correct,
        "reference_frozen": reference_frozen,
        "target_frozen": target_frozen,
        "budget_ramp_correct": ramp_correct,
        "budget_ramp_max_relative_error": max(ramp_errors),
        "hard_surrogate_max_abs_difference": hard_surrogate_difference,
        "energy_violation_max": max(row["energy_violation"] for row in rows),
        "energy_penalty_max": max(row["energy_penalty"] for row in rows),
        "dual_final": rows[-1]["energy_dual"],
        "dual_max_observed": max(row["energy_dual"] for row in rows),
        "aggregate_violation_rows": aggregate_violation_rows,
        "aggregate_violation_note": (
            "energy and violation are separately averaged across four accumulation clips; "
            "a positive mean per-clip ReLU can coexist with mean energy below budget"
        ),
        "first_validation_mse": rows[0]["validation_mse"],
        "last_validation_mse": rows[-1]["validation_mse"],
        "first_representation_skill": rows[0]["validation_representation_skill"],
        "last_representation_skill": rows[-1]["validation_representation_skill"],
        "gradient_norm_min": min(row["gradient_norm"] for row in rows),
        "gradient_norm_max": max(row["gradient_norm"] for row in rows),
        "temporal_gradient_norm_min": min(row["temporal_gradient_norm"] for row in rows),
        "temporal_gradient_norm_max": max(row["temporal_gradient_norm"] for row in rows),
        "mean_rate_min": min(row["mean_rate"] for row in rows),
        "post_ramp_on_activity_min": min(row["hard_active_fraction_on"] for row in post_ramp_rows),
        "post_ramp_off_activity_min": min(row["hard_active_fraction_off"] for row in post_ramp_rows),
        "post_ramp_validation_target_energy_ratios": [
            row["validation_target_energy_ratio"] for row in post_ramp_rows
        ],
        "best_reconstruction_event_steps": [
            row["optimizer_step"] for row in rows if row["best_reconstruction_event"]
        ],
        "best_feasible_event_steps": [
            row["optimizer_step"] for row in rows if row["best_feasible_event"]
        ],
        "best_feasible_behavior_correct": best_feasible_correct,
        "checkpoint": checkpoint_state,
        "resume_passed": resume_passed,
        "gradient_audit": audit,
        "dynamic_rf_status": evaluation["summary"]["dynamic_rf_status"],
        "rgc_type_status": evaluation["summary"]["rgc_type_status"],
        "energy_status": evaluation["summary"]["energy_status"],
        "selected_checkpoint_reason": "best_reconstruction",
        "stage_4_pilot": stage_4_pilot,
    }
    (ARTIFACT_DIR / "stage3_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    stage_status = {
        "stage": 3,
        "status": status,
        "optimizer_steps": rows[-1]["optimizer_step"],
        "validation_count": len(rows),
        "target_budget_established": target is not None,
        "budget_ramp_completed": ramp_correct,
        "best_feasible_behavior_correct": best_feasible_correct,
        "gradient_audit_finite": audit["finite"],
        "formal_training_executed": False,
        "formal_dynamic_rf_executed": False,
        "formal_rgc_typing_executed": False,
        "git_executed": False,
        "code_modified": True,
        "files_modified": ["scripts/run_experiment.py", "tests/test_training_contract.py"],
        "stage_4_pilot": stage_4_pilot,
    }
    (ARTIFACT_DIR / "stage3_status.json").write_text(
        json.dumps(stage_status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stage_status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
