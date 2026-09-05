from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from evaluation.mechanistic_retina.artifacts import write_json
from evaluation.mechanistic_retina.mechanism_artifacts import (
    diagnosis_payload,
    run_payload,
    sha256_file,
    write_metric_csvs,
)
from evaluation.mechanistic_retina.mechanism_diagnosis import (
    LegacyDiagnosisRequest,
    run_legacy_diagnosis,
)
from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismRunConfig,
    MechanismTeacher,
    TeacherName,
    build_teachers,
    load_mechanism_config,
    teacher_preflight,
)
from evaluation.mechanistic_retina.mechanism_run_data import SampledCondition
from evaluation.mechanistic_retina.mechanism_run_types import (
    MechanismRunEvidence,
    ProgressEvent,
)
from evaluation.mechanistic_retina.mechanism_protocol_runs import (
    PhaseRequest,
    SampledRequest,
    run_phase,
    sampled_if_supported,
)
from evaluation.mechanistic_retina.mechanism_scoring import (
    MechanismScore,
    final_case,
    score_runs,
)
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import Candidate0Reference, load_candidate0
from training.mechanistic_retina.stages import MechanisticSeedData, build_seed_data


_SOURCES = (
    "models/mechanistic_retina/pathway_gates.py",
    "models/mechanistic_retina/support_partition.py",
    "models/mechanistic_retina/shared_subunits.py",
    "models/mechanistic_retina/model.py",
    "evaluation/mechanistic_retina/mechanism_identifiability.py",
    "evaluation/mechanistic_retina/structural_ablation.py",
    "evaluation/mechanistic_retina/subspace_overlap.py",
    "scripts/run_mechanism_identifiability.py",
    "configs/mechanism_identifiability.yaml",
)


@dataclass(frozen=True, slots=True)
class ProtocolRequest:
    repo_root: Path
    config_path: Path
    progress: Callable[[ProgressEvent], None]


def run_protocol(request: ProtocolRequest) -> str:
    config = load_mechanism_config(request.config_path)
    output = request.repo_root / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("experiment-config.yaml").write_text(
        request.config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    candidate = load_candidate0(
        request.repo_root / config.candidate0_path,
        usage=config.candidate_teacher_usage,
        reference_candidate_index=config.candidate_teacher_reference_index,
    )
    data = build_seed_data(19, candidate)
    write_json(
        output / "identity-manifest.json",
        _identity_payload(request.repo_root, candidate, data),
    )
    diagnosis = run_legacy_diagnosis(
        LegacyDiagnosisRequest(
            data,
            candidate,
            config,
            lambda model, step, ce: request.progress(
                ProgressEvent("P0", "H1-existing", model, 19, step, ce, 0.0, 0.0)
            ),
        )
    )
    write_json(output / "diagnosis-results.json", diagnosis_payload(diagnosis))
    teachers = build_teachers(data, candidate)
    preflight = teacher_preflight(data, teachers)
    write_json(
        output / "teacher-preflight-results.json",
        {
            name: {
                "passed": result.passed,
                "removal_fraction": result.removal_fraction,
                "pathway_rf_fraction": result.pathway_rf_fraction,
                "heldout_effect": result.heldout_effect,
                "probe_names": list(result.probe_names),
                "probe_effects": list(result.probe_effects),
            }
            for name, result in preflight.items()
        },
    )
    selected = tuple(
        teachers[name.value]
        for name in (TeacherName.BASE, TeacherName.H1, TeacherName.AC)
        if name is TeacherName.BASE or preflight[name.value].passed
    )
    noise_runs = run_phase(
        PhaseRequest(selected, data, candidate, config, None, request.progress)
    )
    noise_score = score_runs(noise_runs, sampled=False)
    write_json(
        output / "noise-free-results.json",
        {
            "score": dict(_score_payload(noise_score)),
            "runs": [dict(run_payload(run)) for run in noise_runs],
        },
    )
    sampled_runs, bank_payload = sampled_if_supported(
        SampledRequest(
            PhaseRequest(selected, data, candidate, config, None, request.progress),
            noise_score,
        )
    )
    sampled_score = score_runs(sampled_runs, sampled=True) if sampled_runs else None
    write_json(
        output / "sampled-confirmation-results.json",
        {
            "executed": bool(sampled_runs),
            "banks": bank_payload,
            "score": None if sampled_score is None else dict(_score_payload(sampled_score)),
            "runs": [dict(run_payload(run)) for run in sampled_runs],
        },
    )
    case = final_case(noise_score, sampled_score)
    write_metric_csvs(output, noise_runs + sampled_runs)
    output.joinpath("decision-report-zh.md").write_text(
        _decision_report(case, diagnosis.decision, noise_score, sampled_score),
        encoding="utf-8",
    )
    request.progress(ProgressEvent("P2", case, "decision", 0, 400, 0.0, 1.0, 1.0))
    return case


def _score_payload(score: MechanismScore) -> Mapping[str, JsonValue]:
    return {
        "h1_passing_seeds": score.h1_passing_seeds,
        "ac_passing_seeds": score.ac_passing_seeds,
        "base_h1_passing_seeds": score.base_h1_passing_seeds,
        "base_ac_passing_seeds": score.base_ac_passing_seeds,
        "rf_passing_runs": score.rf_passing_seeds,
        "h1_passed": score.h1_passed,
        "ac_passed": score.ac_passed,
        "base_passed": score.base_passed,
        "rf_passed": score.rf_passed,
    }


def _identity_payload(
    repo_root: Path,
    candidate: Candidate0Reference,
    data: MechanisticSeedData,
) -> Mapping[str, JsonValue]:
    return {
        "architecture_id": "mechanism_identifiable",
        "initialization": "teacher-independent-raw",
        "parent_checkpoint": None,
        "warm_start_detected": False,
        "retained_T2_seed19_three_bank_rerun": False,
        "lineage_evidence": "direct_model_eval.build_model constructs before train; no checkpoint load",
        "candidate0_sha256": candidate.rf_sha256,
        "candidate_teacher_usage": candidate.teacher_usage.value,
        "candidate_artifact_case": candidate.artifact_case,
        "candidate_selected_index": candidate.selected_candidate,
        "candidate_loaded_index": candidate.loaded_candidate_index,
        "candidate_preflight_passed": candidate.preflight_passed,
        "cell_ids": list(data.cell_ids),
        "final_test_boundary": list(data.final_test_boundary),
        "source_hashes": {path: sha256_file(repo_root / path) for path in _SOURCES},
    }


def _decision_report(
    case: str,
    p0: str,
    noise: MechanismScore,
    sampled: MechanismScore | None,
) -> str:
    sampled_text = (
        "未执行"
        if sampled is None
        else json.dumps(dict(_score_payload(sampled)), ensure_ascii=False)
    )
    return (
        f"# 机制可辨识性结论\n\n- 唯一结论：{case}\n- P0 判定：{p0}\n"
        f"- noise-free：{json.dumps(dict(_score_payload(noise)), ensure_ascii=False)}\n"
        f"- sampled T=2：{sampled_text}\n"
    )


__all__ = ["ProtocolRequest", "run_protocol"]
