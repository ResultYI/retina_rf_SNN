from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from evaluation.mechanistic_retina.artifacts import write_json
from evaluation.mechanistic_retina.mechanism_decision_report import decision_report_zh
from evaluation.mechanistic_retina.mechanism_false_positive import (
    evaluate_false_positives,
    false_positive_payload,
)
from evaluation.mechanistic_retina.mechanism_heldout import HeldoutPathway
from evaluation.mechanistic_retina.mechanism_heldout_artifacts import (
    decide_pathway,
    final_decision,
    heldout_payload,
    write_per_seed_csv,
)
from evaluation.mechanistic_retina.mechanism_heldout_metrics import (
    HeldoutEvaluationRequest,
    HeldoutSeedMetrics,
    evaluate_heldout,
)
from evaluation.mechanistic_retina.mechanism_replay_artifacts import (
    checkpoint_manifest_payload,
    replay_results_payload,
)
from evaluation.mechanistic_retina.mechanism_replay_identity import (
    ReplayContext,
    identity_payload,
    prepare_replay_context,
)
from evaluation.mechanistic_retina.mechanism_replay_runs import (
    find_checkpoint,
    load_checkpoint_model,
    run_minimal_replay,
)
from evaluation.mechanistic_retina.mechanism_replay_types import (
    ReplayExecutionRequest,
    ReplayKey,
    ReplayRunSet,
)
from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    ProgressEvent,
)


@dataclass(frozen=True, slots=True)
class ReplayProtocolRequest:
    repo_root: Path
    run_id: str
    progress: Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class ReplayProtocolError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def run_checkpoint_replay(request: ReplayProtocolRequest) -> str:
    evidence = request.repo_root / ".omo/evidence/mechanism-heldout-final"
    checkpoints = request.repo_root / "runs/mechanism_identifiable_final"
    evidence.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    context = prepare_replay_context(request.repo_root, request.run_id)
    write_json(evidence / "identity-manifest.json", identity_payload(context))
    run_set = run_minimal_replay(
        ReplayExecutionRequest(
            request.repo_root,
            evidence,
            checkpoints,
            context,
            request.progress,
        )
    )
    write_json(evidence / "replay-results.json", replay_results_payload(run_set))
    write_json(
        evidence / "checkpoint-manifest.json",
        checkpoint_manifest_payload(run_set.checkpoints),
    )
    if not run_set.passed():
        case = "SCIENTIFIC-REPLAY-DIVERGED"
        write_json(
            evidence / "failure.json",
            {
                "case": case,
                "reason": "replay exceeded a frozen CE/RF/gate tolerance",
                "heldout_evaluation_executed": False,
            },
        )
        evidence.joinpath("decision-report-zh.md").write_text(
            f"# 最终裁决\n\n唯一最终 Case：`{case}`。已在 held-out evaluation 前停止。\n",
            encoding="utf-8",
        )
        request.progress(
            ProgressEvent("decision", case, "stopped", 0, 400, 0.0, 0.0, 0.0)
        )
        return case
    false_positive = evaluate_false_positives(run_set)
    h1_metrics = _evaluate_pathway(context, run_set, HeldoutPathway.H1)
    ac_metrics = _evaluate_pathway(context, run_set, HeldoutPathway.AC)
    h1 = decide_pathway(HeldoutPathway.H1, h1_metrics, false_positive)
    ac = decide_pathway(HeldoutPathway.AC, ac_metrics, false_positive)
    decision = final_decision(h1, ac, false_positive)
    write_json(evidence / "heldout-h1-results.json", heldout_payload(h1))
    write_json(evidence / "heldout-ac-results.json", heldout_payload(ac))
    write_json(
        evidence / "false-positive-results.json",
        false_positive_payload(false_positive),
    )
    write_per_seed_csv(evidence / "per-seed-metrics.csv", decision)
    evidence.joinpath("decision-report-zh.md").write_text(
        decision_report_zh(decision, len(run_set.checkpoints)),
        encoding="utf-8",
    )
    request.progress(
        ProgressEvent(
            "decision",
            decision.case,
            "heldout",
            0,
            400,
            0.0,
            1.0,
            1.0,
        )
    )
    return decision.case


def _evaluate_pathway(
    context: ReplayContext,
    run_set: ReplayRunSet,
    pathway: HeldoutPathway,
) -> tuple[HeldoutSeedMetrics, ...]:
    match pathway:
        case HeldoutPathway.H1:
            teacher = context.teachers.h1
            teacher_id = "H1-specific"
            structural = AblationName.NO_H1
        case HeldoutPathway.AC:
            teacher = context.teachers.ac
            teacher_id = "AC-specific"
            structural = AblationName.NO_AC
        case unreachable:
            assert_never(unreachable)
    rows = []
    for seed in context.config.seeds:
        full_entry = find_checkpoint(
            run_set, ReplayKey(teacher_id, AblationName.FULL, seed)
        )
        structural_entry = find_checkpoint(
            run_set, ReplayKey(teacher_id, structural, seed)
        )
        metric = evaluate_heldout(
            HeldoutEvaluationRequest(
                pathway,
                teacher,
                load_checkpoint_model(context, full_entry),
                load_checkpoint_model(context, structural_entry),
                context.data,
                structural,
                seed,
            )
        )
        if metric.optimizer_steps != 0:
            raise ReplayProtocolError("held-out evaluator consumed optimizer steps")
        rows.append(metric)
    return tuple(rows)


__all__ = ["ReplayProtocolError", "ReplayProtocolRequest", "run_checkpoint_replay"]
