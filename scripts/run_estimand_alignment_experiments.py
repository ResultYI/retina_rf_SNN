# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "torch"]
# ///
# How to run: python scripts/run_estimand_alignment_experiments.py --help

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.rgc_response import RGCResponseContractError, load_rgc_response
from evaluation.history_estimands import (
    HistoryEstimandError,
    HistoryEstimandRequest,
    audit_history_estimands,
)
from evaluation.response_metrics import compute_response_metrics, training_baseline_rates
from evaluation.teacher_identifiability import (
    TeacherIdentifiabilityError,
    reconstruct_teacher_targets,
)
from evaluation.teacher_self_fit import (
    TeacherSelfFitError,
    TeacherSelfFitRequest,
    audit_teacher_self_fit,
)
from scripts.evaluation.estimand_report_status import (
    exploratory_type_effects as _exploratory_type_effects,
    overall_status as _overall_status,
    parameter_recovery_status as _parameter_recovery_status,
)

SMOKE_MONTE_CARLO_SEEDS = 1
SMOKE_BOOTSTRAP_SEEDS = 1
SMOKE_ITERATIONS = 1
SMOKE_TRIAL_COUNTS = (2,)
DEFAULT_TRIAL_COUNTS = (2, 4, 8, 16, 32, 64)


def main() -> None:
    args = parse_estimand_args(sys.argv[1:])
    result = build_estimand_alignment_report(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_estimand_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher self-fit and history-estimand diagnostics."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-steps", type=int, default=64)
    parser.add_argument("--endogenous-trials", type=int, default=None)
    parser.add_argument("--monte-carlo-seeds", type=int, default=None)
    parser.add_argument("--self-fit-iterations", type=int, default=None)
    parser.add_argument("--self-fit-trial-counts", default=None)
    parser.add_argument("--convergence-check-seeds", type=int, default=None)
    parser.add_argument("--convergence-check-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.endogenous_trials = _resolve(args.endogenous_trials, args.smoke, 64, 2048)
    args.monte_carlo_seeds = _resolve(
        args.monte_carlo_seeds,
        args.smoke,
        SMOKE_MONTE_CARLO_SEEDS,
        20,
    )
    args.self_fit_iterations = _resolve(
        args.self_fit_iterations,
        args.smoke,
        SMOKE_ITERATIONS,
        30,
    )
    args.convergence_check_seeds = _resolve(
        args.convergence_check_seeds,
        args.smoke,
        SMOKE_BOOTSTRAP_SEEDS,
        10,
    )
    args.convergence_check_iterations = _resolve(
        args.convergence_check_iterations,
        args.smoke,
        SMOKE_ITERATIONS,
        100,
    )
    args.self_fit_trial_counts = (
        _parse_trial_counts(args.self_fit_trial_counts)
        if args.self_fit_trial_counts is not None
        else SMOKE_TRIAL_COUNTS
        if args.smoke
        else DEFAULT_TRIAL_COUNTS
    )
    return args


def build_estimand_alignment_report(args: argparse.Namespace):
    observational = _observational_response(args)
    parameter = _parameter_recovery(args)
    exploratory = _exploratory_type_effects(parameter)
    return {
        "parameter_recovery": parameter,
        "observational_response": observational,
        "exploratory_type_effects": exploratory,
        "status": _overall_status(parameter["status"], observational["status"]),
    }


def _parameter_recovery(args: argparse.Namespace):
    try:
        history = audit_history_estimands(
            HistoryEstimandRequest(
                args.validation,
                probe_steps=args.probe_steps,
                endogenous_trials=args.endogenous_trials,
                seed=args.seed,
            )
        )
        self_fit = audit_teacher_self_fit(
            TeacherSelfFitRequest(
                args.validation,
                trial_counts=args.self_fit_trial_counts,
                monte_carlo_seeds=args.monte_carlo_seeds,
                max_iterations=args.self_fit_iterations,
                probe_steps=args.probe_steps,
                seed=args.seed,
                device=args.device,
            )
        )
        convergence_check = audit_teacher_self_fit(
            TeacherSelfFitRequest(
                args.validation,
                trial_counts=(max(args.self_fit_trial_counts),),
                monte_carlo_seeds=args.convergence_check_seeds,
                max_iterations=args.convergence_check_iterations,
                probe_steps=args.probe_steps,
                seed=args.seed,
                device=args.device,
            )
        )
    except (
        HistoryEstimandError,
        RGCResponseContractError,
        TeacherIdentifiabilityError,
        TeacherSelfFitError,
    ) as exc:
        return {
            "status": "not_identifiable",
            "reason": str(exc),
            "history_contracts": ("zero", "matched_observed", "standard_train_rate"),
        }
    point = self_fit.points[-1]
    controlled_results = {
        key: asdict(value) for key, value in history.controlled_by_history.items()
    }
    status = _parameter_recovery_status(
        controlled_results,
        per_cell_kernel_correlation=point.per_cell_kernel_correlation,
        per_cell_kernel_error=point.per_cell_kernel_error,
    )
    return {
        "status": status,
        "history_contracts": ("zero", "matched_observed", "standard_train_rate"),
        "controlled_history_results": controlled_results,
        "per_cell_kernel_error": point.per_cell_kernel_error,
        "per_cell_kernel_correlation": point.per_cell_kernel_correlation,
        "group_effect_recovery": tuple(asdict(effect) for effect in point.group_effect_recovery),
        "cross_seed_stability": {
            "seed_count": self_fit.monte_carlo_seeds,
            "per_cell_context_kernel_correlation": point.per_cell_context_kernel_correlation,
        },
        "controlled_history_diagnostic": asdict(history),
        "self_fit_power": asdict(self_fit),
        "self_fit_convergence_check": asdict(convergence_check),
    }


def _observational_response(args: argparse.Namespace):
    session = load_rgc_response(args.validation)
    targets = torch.as_tensor(session.spike_counts)
    mask = torch.as_tensor(session.valid_mask)
    baseline = training_baseline_rates(targets, mask)
    try:
        probabilities = reconstruct_teacher_targets(args.validation).conditional_probabilities
        logits = torch.logit(torch.as_tensor(probabilities).clamp(1e-5, 1 - 1e-5))
    except (TeacherIdentifiabilityError, RGCResponseContractError):
        logits = torch.logit(baseline.clamp(1e-5, 1 - 1e-5)).view(1, 1, 1, -1)
        logits = logits.expand_as(targets)
    metrics = compute_response_metrics(
        logits,
        targets,
        mask,
        session.target_kind,
        baseline,
    )
    decomposition = _history_decomposition(args)
    return {
        "status": "supported",
        "metrics": asdict(metrics),
        "history_mediated_decomposition": decomposition,
    }


def _history_decomposition(args: argparse.Namespace):
    try:
        history = audit_history_estimands(
            HistoryEstimandRequest(
                args.validation,
                probe_steps=args.probe_steps,
                endogenous_trials=args.endogenous_trials,
                seed=args.seed,
            )
        )
    except (HistoryEstimandError, RGCResponseContractError, TeacherIdentifiabilityError) as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "direct_response_gain": history.direct_response_gain,
        "history_mediated_gain": history.history_mediated_gain,
        "marginal_response_gain": history.marginal_response_gain,
        "sign_inversion_cell_ids": history.sign_inversion_cell_ids,
        "decomposition_residual": history.decomposition_residual,
    }


def _resolve(value: int | None, smoke: bool, smoke_default: int, default: int) -> int:
    if value is not None:
        return value
    return smoke_default if smoke else default


def _parse_trial_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(",") if part)
    if not counts:
        raise argparse.ArgumentTypeError("At least one trial count is required")
    return counts


if __name__ == "__main__":
    main()
