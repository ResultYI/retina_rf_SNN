from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch.nn import functional as F

from baselines.point_process_glm import PointProcessGLM
from evaluation.mechanistic_retina.direct_metrics import DirectRFSummary, rf_summary
from evaluation.mechanistic_retina.rf_base import Candidate0Reference


LogitFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class CellIdentityResult:
    cell_id: str
    nearest_cell_id: str
    nearest_type_polarity_resolved: bool
    prototype_centroid_resolved: bool


@dataclass(frozen=True, slots=True)
class RFComparison:
    summary: DirectRFSummary
    nearest_type_polarity_fraction: float
    prototype_centroid_fraction: float
    identities: tuple[CellIdentityResult, ...]

    @property
    def exact_fraction(self) -> float:
        return self.summary.metric.exact_fraction


def conditional_total_dynamic_rf(
    logits: LogitFunction,
    cones: torch.Tensor,
    observed_counts: torch.Tensor,
    lag_steps: int = 16,
) -> torch.Tensor:
    stimulus = cones.detach().clone().requires_grad_(True)
    final_logits = logits(stimulus, observed_counts)[:, -1]
    rows = []
    for cell in range(final_logits.shape[-1]):
        gradient = torch.autograd.grad(
            final_logits[:, cell].sum(),
            stimulus,
            retain_graph=cell + 1 < final_logits.shape[-1],
        )[0]
        rows.append(gradient[:, -lag_steps:])
    return torch.stack(rows, dim=1).detach()


def glm_filter_rf(model: PointProcessGLM, contexts: int) -> torch.Tensor:
    chronological = model.kernel.detach().flip(1)
    return chronological.unsqueeze(0).expand(contexts, -1, -1, -1).clone()


def evaluate_comparison_rf(
    predicted: torch.Tensor,
    candidate: Candidate0Reference,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
) -> RFComparison:
    summary = rf_summary(
        predicted,
        candidate.rf,
        cone_positions,
        cell_positions,
        candidate.metadata,
    )
    nearest = _nearest_teacher(predicted, candidate.rf)
    identities = tuple(
        CellIdentityResult(
            metadata.cell_id,
            candidate.metadata[int(nearest[index])].cell_id,
            _same_type_polarity(metadata, candidate.metadata[int(nearest[index])]),
            bool(summary.metric.cells[index].type_polarity_resolved),
        )
        for index, metadata in enumerate(candidate.metadata)
    )
    return RFComparison(
        summary,
        sum(value.nearest_type_polarity_resolved for value in identities) / len(identities),
        summary.metric.type_polarity_fraction,
        identities,
    )


def rf_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.flatten().double(), second.flatten().double(), dim=0))


def _nearest_teacher(predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    contexts = predicted.shape[0]
    expanded = teacher.unsqueeze(0).expand(contexts, -1, -1, -1)
    left = predicted.permute(1, 0, 2, 3).reshape(predicted.shape[1], -1).double()
    right = expanded.permute(1, 0, 2, 3).reshape(teacher.shape[0], -1).double()
    similarity = left @ right.T
    similarity /= (left.norm(dim=1, keepdim=True) * right.norm(dim=1)[None]).clamp_min(1e-12)
    return similarity.argmax(dim=1)


def _same_type_polarity(first, second) -> bool:
    return first.type_id == second.type_id and first.polarity == second.polarity


__all__ = [
    "CellIdentityResult",
    "RFComparison",
    "conditional_total_dynamic_rf",
    "evaluate_comparison_rf",
    "glm_filter_rf",
    "rf_cosine",
]
