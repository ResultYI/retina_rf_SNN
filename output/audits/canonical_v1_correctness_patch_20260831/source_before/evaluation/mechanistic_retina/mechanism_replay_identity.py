from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.mechanism_checkpoints import (
    sha256_file,
    tensors_sha256,
)
from evaluation.mechanistic_retina.mechanism_identifiability import (
    MechanismTeacher,
    TeacherFamily,
    build_teachers,
    teacher_preflight,
)
from evaluation.mechanistic_retina.mechanism_runtime import (
    MechanismRunConfig,
    load_mechanism_config,
)
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import Candidate0Reference, load_candidate0
from training.mechanistic_retina.stages import MechanisticSeedData, build_seed_data


_PRIOR_EVIDENCE = Path(".omo/evidence/mechanism-identifiable-retina")
_CONFIG = Path("configs/mechanism_identifiability.yaml")


@dataclass(frozen=True, slots=True)
class ReplayIdentity:
    run_id: str
    architecture_revision: str
    prior_identity_matched: bool
    config_hash: str
    source_hash: str
    source_hashes: Mapping[str, str]
    dataset_identity: str
    teacher_hashes: Mapping[str, str]
    cell_order: tuple[str, ...]
    cone_order: tuple[int, ...]
    lag_order: tuple[int, ...]
    final_test_boundary: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayContext:
    config: MechanismRunConfig
    config_snapshot: Mapping[str, JsonValue]
    candidate: Candidate0Reference
    data: MechanisticSeedData
    teachers: TeacherFamily
    identity: ReplayIdentity


@dataclass(frozen=True, slots=True)
class ReplayIdentityError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return f"SCIENTIFIC_REPLAY_IDENTITY_MISMATCH: {self.message}"


def prepare_replay_context(repo_root: Path, run_id: str) -> ReplayContext:
    config_path = repo_root / _CONFIG
    prior_root = repo_root / _PRIOR_EVIDENCE
    config_snapshot = _read_json(config_path)
    prior_config = _read_json(prior_root / "experiment-config.yaml")
    if config_snapshot != prior_config:
        raise ReplayIdentityError("frozen config differs from prior evidence")
    config = load_mechanism_config(config_path)
    prior_identity = _read_json(prior_root / "identity-manifest.json")
    source_hashes = _verify_sources(repo_root, prior_identity)
    candidate = load_candidate0(
        repo_root / config.candidate0_path,
        usage=config.candidate_teacher_usage,
        reference_candidate_index=config.candidate_teacher_reference_index,
    )
    data = build_seed_data(19, candidate)
    _verify_dataset_anchors(candidate, data, prior_identity)
    teachers = build_teachers(data, candidate)
    _verify_teacher_preflight(data, teachers, prior_root)
    identity = ReplayIdentity(
        run_id,
        "mechanism_identifiable",
        True,
        sha256_file(config_path),
        _mapping_hash(source_hashes),
        source_hashes,
        _dataset_hash(data),
        {teacher.name.value: _teacher_hash(teacher) for teacher in _teachers(teachers)},
        data.cell_ids,
        tuple(range(data.cone_positions.shape[0])),
        tuple(range(16)),
        data.final_test_boundary,
    )
    return ReplayContext(config, config_snapshot, candidate, data, teachers, identity)


def identity_payload(context: ReplayContext) -> Mapping[str, JsonValue]:
    identity = context.identity
    return {
        "schema": "mechanism-heldout-final-identity-v1",
        "run_id": identity.run_id,
        "architecture_revision": identity.architecture_revision,
        "prior_identity_matched": identity.prior_identity_matched,
        "config_hash": identity.config_hash,
        "source_hash": identity.source_hash,
        "source_hashes": dict(identity.source_hashes),
        "dataset_identity": identity.dataset_identity,
        "teacher_hashes": dict(identity.teacher_hashes),
        "prior_teacher_hash_available": False,
        "teacher_identity_verification": "frozen-source-hash-and-exact-preflight",
        "cell_order": list(identity.cell_order),
        "cone_order": list(identity.cone_order),
        "lag_order": list(identity.lag_order),
        "lag_order_semantics": "oldest_to_current",
        "seeds": list(context.config.seeds),
        "optimizer": "Adam",
        "learning_rate": context.config.learning_rate,
        "batch_size": context.config.batch_size,
        "steps": context.config.steps,
        "checkpoints": list(context.config.checkpoints),
        "initialization": "teacher-independent-raw",
        "parent_checkpoint": None,
        "support_partition_hash": identity.source_hashes[
            "models/mechanistic_retina/support_partition.py"
        ],
        "gate_definition_hash": identity.source_hashes[
            "models/mechanistic_retina/pathway_gates.py"
        ],
        "final_test_boundary": list(identity.final_test_boundary),
        "final_test_scientific_evaluation_consumed": False,
    }


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReplayIdentityError(f"cannot read identity input: {path}") from error


def _verify_sources(repo_root: Path, prior_identity) -> Mapping[str, str]:
    expected = prior_identity["source_hashes"]
    actual = {name: sha256_file(repo_root / name) for name in expected}
    if actual != expected:
        changed = sorted(name for name in expected if actual.get(name) != expected[name])
        raise ReplayIdentityError(f"frozen source hash changed: {changed}")
    return actual


def _verify_dataset_anchors(
    candidate: Candidate0Reference,
    data: MechanisticSeedData,
    prior_identity,
) -> None:
    if candidate.rf_sha256 != prior_identity["candidate0_sha256"]:
        raise ReplayIdentityError("Candidate0 hash changed")
    if list(data.cell_ids) != prior_identity["cell_ids"]:
        raise ReplayIdentityError("cell order changed")
    if list(data.final_test_boundary) != prior_identity["final_test_boundary"]:
        raise ReplayIdentityError("final-test boundary changed")


def _verify_teacher_preflight(
    data: MechanisticSeedData,
    teachers: TeacherFamily,
    prior_root: Path,
) -> None:
    previous = _read_json(prior_root / "teacher-preflight-results.json")
    current = teacher_preflight(data, teachers)
    for name, result in current.items():
        expected = previous[name]
        if result.passed is not expected["passed"]:
            raise ReplayIdentityError(f"teacher preflight status changed: {name}")
        if list(result.probe_names) != expected["probe_names"]:
            raise ReplayIdentityError(f"teacher probe names changed: {name}")
        if list(result.probe_effects) != expected["probe_effects"]:
            raise ReplayIdentityError(f"teacher probe effects changed: {name}")
        values = (
            (result.removal_fraction, expected["removal_fraction"]),
            (result.pathway_rf_fraction, expected["pathway_rf_fraction"]),
            (result.heldout_effect, expected["heldout_effect"]),
        )
        if any(current_value != expected_value for current_value, expected_value in values):
            raise ReplayIdentityError(f"teacher preflight metric changed: {name}")


def _dataset_hash(data: MechanisticSeedData) -> str:
    tensor_hash = tensors_sha256(
        {
            "train_cones": data.train_cones,
            "train_probability": data.train_probability,
            "train_mask": data.train_mask,
            "validation_cones": data.validation_cones,
            "validation_probability": data.validation_probability,
            "validation_mask": data.validation_mask,
            "cone_positions": data.cone_positions,
            "cell_positions": data.cell_positions,
        }
    )
    metadata = {
        "seed": data.seed,
        "cell_ids": data.cell_ids,
        "cell_types": data.cell_types,
        "polarities": data.polarities,
        "final_test_boundary": data.final_test_boundary,
        "tensor_hash": tensor_hash,
    }
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _teacher_hash(teacher: MechanismTeacher) -> str:
    tensors = dict(teacher.model.state_dict())
    tensors.update(
        {
            "teacher.train_probability": teacher.train_probability,
            "teacher.validation_probability": teacher.validation_probability,
            "teacher.bc_rf": teacher.bc_rf,
            "teacher.pathway_rf": teacher.pathway_rf,
            "teacher.total_rf": teacher.total_rf,
            "teacher.response_bias": teacher.response_bias,
        }
    )
    return tensors_sha256(tensors)


def _mapping_hash(values: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(values), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _teachers(family: TeacherFamily) -> tuple[MechanismTeacher, ...]:
    return family.base, family.h1, family.ac


__all__ = [
    "ReplayContext",
    "ReplayIdentity",
    "ReplayIdentityError",
    "identity_payload",
    "prepare_replay_context",
]
