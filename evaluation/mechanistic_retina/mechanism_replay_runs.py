from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.artifacts import write_json
from evaluation.mechanistic_retina.mechanism_artifacts import run_payload
from evaluation.mechanistic_retina.mechanism_checkpoints import (
    CheckpointIdentity,
    load_final_checkpoint,
    save_final_checkpoint,
)
from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismTeacher,
    TeacherName,
    build_student,
)
from evaluation.mechanistic_retina.mechanism_replay_identity import ReplayContext
from evaluation.mechanistic_retina.mechanism_replay_artifacts import (
    checkpoint_manifest_payload,
)
from evaluation.mechanistic_retina.mechanism_replay_types import (
    CheckpointManifestEntry,
    ReplayExecutionRequest,
    ReplayKey,
    ReplayMetricComparison,
    ReplayRunError,
    ReplayRunSet,
)
from evaluation.mechanistic_retina.mechanism_run_types import (
    AblationName,
    MechanismRunEvidence,
    ProgressEvent,
    TeacherRunRequest,
)
from evaluation.mechanistic_retina.mechanism_runs import run_teacher_ablation
from evaluation.mechanistic_retina.metrics import JsonValue
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def run_minimal_replay(request: ReplayExecutionRequest) -> ReplayRunSet:
    previous = _previous_runs(request.repo_root)
    runs = []
    checkpoints = []
    comparisons = []
    for seed in request.context.config.seeds:
        for teacher, variants in _matrix(request.context):
            for variant in variants:
                run = run_teacher_ablation(
                    TeacherRunRequest(
                        teacher,
                        request.context.data,
                        request.context.candidate,
                        request.context.config,
                        seed,
                        None,
                        request.progress,
                    ),
                    variant,
                )
                entry = _persist_run(request, run, teacher)
                current = run_payload(run)
                comparison = replay_metric_comparison(
                    current,
                    previous[ReplayKey(run.teacher, run.ablation, run.seed)],
                )
                runs.append(run)
                checkpoints.append(entry)
                comparisons.append(comparison)
                write_json(
                    request.evidence_dir / "checkpoint-manifest.json",
                    checkpoint_manifest_payload(tuple(checkpoints)),
                )
                request.progress(
                    ProgressEvent(
                        "checkpoint",
                        run.teacher,
                        run.ablation.value,
                        run.seed,
                        request.context.config.steps,
                        run.validation_ce,
                        run.rf.metric.global_cosine,
                        max(run.gates.h1, run.gates.ac_local, run.gates.ac_transient),
                    )
                )
    return ReplayRunSet(tuple(runs), tuple(checkpoints), tuple(comparisons))


def replay_metric_comparison(
    current: Mapping[str, JsonValue],
    previous: Mapping[str, JsonValue],
) -> ReplayMetricComparison:
    current_gates = current["gates"]
    previous_gates = previous["gates"]
    current_paths = current["pathway_cosines"]
    previous_paths = previous["pathway_cosines"]
    current_rf = current["rf"]
    previous_rf = previous["rf"]
    ce = abs(float(current["validation_ce"]) - float(previous["validation_ce"]))
    gate = max(
        abs(float(current_gates[name]) - float(previous_gates[name]))
        for name in ("h1", "ac_local", "ac_transient", "history")
    )
    pathway = max(
        abs(float(current_paths[name]) - float(previous_paths[name]))
        for name in ("BC", "H1", "AC")
    )
    total = abs(
        float(current_rf["global_cosine"]) - float(previous_rf["global_cosine"])
    )
    exact = abs(
        float(current_rf["exact_fraction"]) - float(previous_rf["exact_fraction"])
    )
    return ReplayMetricComparison(
        ce,
        gate,
        pathway,
        total,
        exact,
        ce <= 1e-4
        and gate <= 1e-3
        and pathway <= 1e-4
        and total <= 1e-4
        and exact == 0.0,
    )


def load_checkpoint_model(
    context: ReplayContext,
    entry: CheckpointManifestEntry,
) -> MechanisticGraphTemporalRetina:
    model = build_student(context.data, entry.saved.identity.seed)
    identity = load_final_checkpoint(entry.saved.path, model)
    if identity != entry.saved.identity:
        raise ReplayRunError(f"checkpoint identity changed: {entry.relative_path}")
    return model


def find_checkpoint(
    run_set: ReplayRunSet,
    key: ReplayKey,
) -> CheckpointManifestEntry:
    for entry in run_set.checkpoints:
        identity = entry.saved.identity
        if (
            identity.teacher_id == key.teacher
            and identity.structural_variant is key.variant
            and identity.seed == key.seed
        ):
            return entry
    raise ReplayRunError(
        f"missing checkpoint: {key.teacher}/{key.variant.value}/seed-{key.seed}"
    )


def _persist_run(
    request: ReplayExecutionRequest,
    run: MechanismRunEvidence,
    teacher: MechanismTeacher,
) -> CheckpointManifestEntry:
    identity = _checkpoint_identity(request.context, run, teacher)
    path = _checkpoint_path(
        request.checkpoint_root,
        ReplayKey(teacher.name.value, run.ablation, run.seed),
    )
    saved = save_final_checkpoint(path, run.model, identity)
    restored = build_student(request.context.data, run.seed)
    loaded = load_final_checkpoint(path, restored)
    equal = loaded == identity and all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in run.model.state_dict().items()
    )
    gate_difference = max(
        abs(float(restored.state_dict()[name]) - float(run.model.state_dict()[name]))
        for name in (
            "gates.raw_h1_amplitude",
            "gates.ac_local",
            "gates.ac_transient",
            "gates.history",
        )
    )
    return CheckpointManifestEntry(
        saved,
        path.relative_to(request.repo_root).as_posix(),
        equal,
        gate_difference,
    )


def _checkpoint_identity(
    context: ReplayContext,
    run: MechanismRunEvidence,
    teacher: MechanismTeacher,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        context.identity.architecture_revision,
        teacher.name.value,
        context.identity.teacher_hashes[teacher.name.value],
        _condition(teacher.name),
        run.ablation,
        run.seed,
        context.config.steps,
        context.identity.run_id,
        context.identity.dataset_identity,
        context.identity.cell_order,
        context.identity.cone_order,
        context.identity.lag_order,
        {
            "h1": run.gates.h1,
            "ac_local": run.gates.ac_local,
            "ac_transient": run.gates.ac_transient,
            "history": run.gates.history,
        },
        context.config_snapshot,
        context.identity.config_hash,
        context.identity.source_hash,
    )


def _previous_runs(repo_root: Path):
    path = repo_root / ".omo/evidence/mechanism-identifiable-retina/noise-free-results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        ReplayKey(
            str(run["teacher"]),
            AblationName(str(run["model"])),
            int(run["seed"]),
        ): run
        for run in payload["runs"]
    }


def _matrix(context: ReplayContext):
    return (
        (context.teachers.base, (AblationName.FULL,)),
        (context.teachers.h1, (AblationName.FULL, AblationName.NO_H1)),
        (context.teachers.ac, (AblationName.FULL, AblationName.NO_AC)),
    )


def _condition(name: TeacherName) -> str:
    return {
        TeacherName.BASE: "base",
        TeacherName.H1: "h1",
        TeacherName.AC: "ac",
    }[name]


def _checkpoint_path(
    root: Path,
    key: ReplayKey,
) -> Path:
    filename = key.variant.value.lower().replace("-", "_") + ".pt"
    return root / _condition(TeacherName(key.teacher)) / f"seed-{key.seed}" / filename


__all__ = [
    "CheckpointManifestEntry",
    "ReplayExecutionRequest",
    "ReplayMetricComparison",
    "ReplayRunError",
    "ReplayRunSet",
    "find_checkpoint",
    "load_checkpoint_model",
    "replay_metric_comparison",
    "run_minimal_replay",
]
