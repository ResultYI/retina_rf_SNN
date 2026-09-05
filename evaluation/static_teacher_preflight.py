from __future__ import annotations

from dataclasses import dataclass
import statistics

import numpy as np
import torch

from evaluation.static_teacher import (
    StaticTeacherCandidate,
    StaticTeacherCellParameter,
    StaticTeacherConfig,
    build_static_teacher_candidate,
)
from evaluation.v4_group_endpoint import GroupEndpointAudit
from evaluation.v4_group_endpoint import GroupEndpointRequest, audit_group_endpoint
from evaluation.v4_group_scoring import GroupEndpointKind
from evaluation.v4_identity_endpoint import TeacherIdentityAudit
from evaluation.v4_identity_endpoint import audit_teacher_identity


@dataclass(frozen=True, slots=True)
class TeacherPreflightResult:
    candidate_index: int
    exact_identity: TeacherIdentityAudit
    type_polarity: GroupEndpointAudit
    type_collapsed_location: GroupEndpointAudit
    median_exact_margin: float
    full_glm_rf_cosine: float
    rf_norms: tuple[float, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class StaticTeacherPreflightError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def noise_free_glm_rf_cosine(
    rf: torch.Tensor,
    *,
    seed: int = 73001,
) -> float:
    if rf.ndim != 5 or rf.shape[0] != 1 or rf.shape[1] != 1:
        raise StaticTeacherPreflightError("preflight RF must have one static context")
    cell_rf = rf[0, 0].double().numpy()
    cell_count, lag_count, cone_count = cell_rf.shape
    rng = np.random.default_rng(seed)
    stimulus = rng.normal(size=(112, lag_count + 8, cone_count))
    design = np.concatenate(
        [
            stimulus[:, time - lag_count + 1 : time + 1].reshape(112, -1)
            for time in range(lag_count - 1, stimulus.shape[1])
        ]
    )
    logits = design @ cell_rf.reshape(cell_count, -1).T
    recovered, _, _, _ = np.linalg.lstsq(design, logits, rcond=None)
    estimated = recovered.T.reshape(cell_rf.shape)
    denominator = np.linalg.norm(estimated) * np.linalg.norm(cell_rf)
    return float(np.vdot(estimated, cell_rf) / max(denominator, 1e-12))


def preflight_static_teacher(candidate: StaticTeacherCandidate) -> TeacherPreflightResult:
    exact = audit_teacher_identity(candidate.rf, candidate.metadata)
    type_polarity = audit_group_endpoint(
        GroupEndpointRequest(
            candidate.rf,
            candidate.metadata,
            GroupEndpointKind.TYPE_POLARITY,
        )
    )
    collapsed = audit_group_endpoint(
        GroupEndpointRequest(
            candidate.rf,
            candidate.metadata,
            GroupEndpointKind.TYPE_COLLAPSED_POLARITY_LOCATION,
        )
    )
    margins = tuple(cell.intrinsic_margin for cell in exact.cells)
    median = float(statistics.median(margins))
    glm_cosine = noise_free_glm_rf_cosine(candidate.rf)
    norms = tuple(
        float(value)
        for value in candidate.rf.movedim(2, 0).reshape(16, -1).norm(dim=1)
    )
    passed = (
        1.0 - exact.ambiguous_fraction >= 0.75
        and type_polarity.group_resolved_fraction is not None
        and type_polarity.group_resolved_fraction >= 0.75
        and collapsed.group_resolved_fraction is not None
        and collapsed.group_resolved_fraction >= 0.75
        and median >= 0.01
        and glm_cosine >= 0.90
        and min(norms) > 1e-8
    )
    return TeacherPreflightResult(
        candidate.config.candidate_index,
        exact,
        type_polarity,
        collapsed,
        median,
        glm_cosine,
        norms,
        passed,
    )


__all__ = [
    "StaticTeacherCandidate",
    "StaticTeacherCellParameter",
    "StaticTeacherConfig",
    "TeacherPreflightResult",
    "build_static_teacher_candidate",
    "noise_free_glm_rf_cosine",
    "preflight_static_teacher",
]
