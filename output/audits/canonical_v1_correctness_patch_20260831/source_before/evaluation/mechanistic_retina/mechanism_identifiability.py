from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum, unique
import math
from typing import assert_never

import torch

from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.mechanistic_retina.mechanism_teacher_support import (
    ProbeRequest,
    matched_bias,
    probe_effects,
    set_teacher_parameters,
)
from evaluation.mechanistic_retina.mechanism_runtime import (
    MechanismRunConfig,
    build_student,
    load_mechanism_config,
    pathway_rfs,
)
from models.mechanistic_retina.contracts import (
    PathwayClamp,
)
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.stages import MechanisticSeedData


@unique
class TeacherName(StrEnum):
    BASE = "Base"
    H1 = "H1-specific"
    AC = "AC-specific"


@dataclass(frozen=True, slots=True)
class MechanismTeacher:
    name: TeacherName
    model: MechanisticGraphTemporalRetina
    train_probability: torch.Tensor
    validation_probability: torch.Tensor
    bc_rf: torch.Tensor
    pathway_rf: torch.Tensor
    total_rf: torch.Tensor
    response_bias: torch.Tensor


@dataclass(frozen=True, slots=True)
class TeacherFamily:
    base: MechanismTeacher
    h1: MechanismTeacher
    ac: MechanismTeacher

    def __iter__(self) -> Iterator[str]:
        return iter((TeacherName.BASE.value, TeacherName.H1.value, TeacherName.AC.value))

    def __getitem__(self, name: str) -> MechanismTeacher:
        match TeacherName(name):
            case TeacherName.BASE:
                return self.base
            case TeacherName.H1:
                return self.h1
            case TeacherName.AC:
                return self.ac
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class TeacherPreflight:
    passed: bool
    removal_fraction: float
    pathway_rf_fraction: float
    heldout_effect: float
    probe_names: tuple[str, ...]
    probe_effects: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TeacherBuildRequest:
    name: TeacherName
    model: MechanisticGraphTemporalRetina
    data: MechanisticSeedData
    target_rate: torch.Tensor


@dataclass(frozen=True, slots=True)
class TeacherBuildError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def build_teachers(
    data: MechanisticSeedData,
    candidate: Candidate0Reference,
) -> TeacherFamily:
    if candidate.rf.shape[0] != len(data.cell_ids):
        raise TeacherBuildError("Candidate0 and mechanism teacher identities differ")
    base_model = build_student(data, 701)
    set_teacher_parameters(base_model)
    with torch.no_grad():
        for gate in base_model.gates.parameters():
            gate.zero_()
        base_model.gates.raw_h1_amplitude.fill_(
            torch.finfo(base_model.gates.raw_h1_amplitude.dtype).min
        )
    models = {
        TeacherName.BASE: deepcopy(base_model),
        TeacherName.H1: deepcopy(base_model),
        TeacherName.AC: deepcopy(base_model),
    }
    with torch.no_grad():
        models[TeacherName.H1].gates.set_h1_amplitude_(0.01)
        models[TeacherName.AC].gates.ac_local.fill_(1.0)
        models[TeacherName.AC].gates.ac_transient.fill_(1.0)
    target_rate = _masked_rate(data.train_probability[:, 0], data.train_mask[:, 0])
    teachers = tuple(
        _build_teacher(TeacherBuildRequest(name, models[name], data, target_rate))
        for name in (TeacherName.BASE, TeacherName.H1, TeacherName.AC)
    )
    return TeacherFamily(teachers[0], teachers[1], teachers[2])


def teacher_preflight(
    data: MechanisticSeedData,
    teachers: TeacherFamily,
) -> Mapping[str, TeacherPreflight]:
    return {
        TeacherName.H1.value: _preflight_one(data, teachers.h1, PathwayClamp.H1),
        TeacherName.AC.value: _preflight_one(data, teachers.ac, PathwayClamp.AMACRINE_LOCAL),
    }


def _build_teacher(
    request: TeacherBuildRequest,
) -> MechanismTeacher:
    train_history = torch.zeros_like(request.data.train_probability[:, 0])
    validation_history = torch.zeros_like(request.data.validation_probability[:, 0])
    with torch.no_grad():
        request.model.rgc.response_bias.zero_()
        train_logits = request.model.forward_sequence(
            request.data.train_cones, observed_counts=train_history
        ).logits
        bias = matched_bias(
            train_logits,
            request.data.train_mask[:, 0],
            request.target_rate,
        )
        request.model.rgc.response_bias.copy_(bias)
        train = torch.sigmoid(train_logits + bias)
        validation = request.model.forward_sequence(
            request.data.validation_cones, observed_counts=validation_history
        ).spike_probability
    paths = pathway_rfs(
        request.model,
        request.data.validation_cones[:2],
        validation_history[:2],
    )
    return MechanismTeacher(
        request.name,
        request.model,
        train,
        validation,
        paths["BC"],
        paths["H1"] if request.name is TeacherName.H1 else paths["AC"],
        sum(paths.values(), torch.zeros_like(paths["BC"])),
        bias.detach().clone(),
    )


def _preflight_one(
    data: MechanisticSeedData,
    teacher: MechanismTeacher,
    clamp: PathwayClamp,
) -> TeacherPreflight:
    clamps = (
        frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})
        if clamp is PathwayClamp.AMACRINE_LOCAL
        else frozenset({clamp})
    )
    history = torch.zeros_like(teacher.validation_probability)
    with torch.no_grad():
        full_logits = teacher.model.forward_sequence(
            data.validation_cones, observed_counts=history
        ).logits
        ablated_logits = teacher.model.forward_sequence(
            data.validation_cones, observed_counts=history, clamps=clamps
        ).logits
    full_ce = float(expected_bernoulli_nll(full_logits, teacher.validation_probability, data.validation_mask[:, 0]))
    ablated_ce = float(expected_bernoulli_nll(ablated_logits, teacher.validation_probability, data.validation_mask[:, 0]))
    bias = torch.logit(_masked_rate(teacher.train_probability, data.train_mask[:, 0]).clamp(1e-6, 1 - 1e-6))
    bias_ce = float(expected_bernoulli_nll(bias.view(1, 1, -1).expand_as(full_logits), teacher.validation_probability, data.validation_mask[:, 0]))
    removal = (ablated_ce - full_ce) / max(1e-12, bias_ce - full_ce)
    rf_fraction = float(teacher.pathway_rf.norm()) / max(1e-12, float(teacher.total_rf.norm()))
    heldout = float((full_logits - ablated_logits).abs().mean())
    names, effects = probe_effects(
        ProbeRequest(
            teacher.model,
            data,
            clamps,
            teacher.name is TeacherName.H1,
        )
    )
    passed = removal >= 0.20 and rf_fraction >= 0.10 and heldout > 0 and all(math.isfinite(value) and value > 0 for value in effects)
    return TeacherPreflight(passed, removal, rf_fraction, heldout, names, effects)


def _masked_rate(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (probability * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)


__all__ = [
    "MechanismRunConfig",
    "MechanismTeacher",
    "TeacherFamily",
    "TeacherName",
    "TeacherPreflight",
    "build_student",
    "build_teachers",
    "load_mechanism_config",
    "pathway_rfs",
    "teacher_preflight",
]
