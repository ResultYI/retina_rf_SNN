from __future__ import annotations

from dataclasses import dataclass

import torch


class RFMapAgreementError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RFMapAgreement:
    center_sign_match: bool
    centroid_distance_degs: float
    cosine_similarity: float


def compare_rf_maps(
    first: torch.Tensor,
    second: torch.Tensor,
    positions_degs: torch.Tensor,
) -> RFMapAgreement:
    if first.shape != second.shape or first.ndim not in {1, 2}:
        raise RFMapAgreementError("RF maps must share shape [cone] or [time,cone]")
    if positions_degs.shape != (first.shape[-1], 2):
        raise RFMapAgreementError("RF positions must have shape [cone,2]")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (first, second, positions_degs)
    ):
        raise RFMapAgreementError("RF agreement inputs must be finite")
    first_spatial = _dominant_spatial_map(first)
    second_spatial = _dominant_spatial_map(second)
    if first_spatial.abs().sum() == 0 or second_spatial.abs().sum() == 0:
        raise RFMapAgreementError("RF maps must contain non-zero structure")
    first_center = int(first_spatial.abs().argmax())
    second_center = int(second_spatial.abs().argmax())
    first_centroid = _centroid(first_spatial, positions_degs)
    second_centroid = _centroid(second_spatial, positions_degs)
    similarity = torch.nn.functional.cosine_similarity(
        first.flatten(),
        second.flatten(),
        dim=0,
    )
    return RFMapAgreement(
        center_sign_match=bool(
            torch.sign(first_spatial[first_center])
            == torch.sign(second_spatial[second_center])
        ),
        centroid_distance_degs=float(
            torch.linalg.vector_norm(first_centroid - second_centroid)
        ),
        cosine_similarity=float(similarity),
    )


def _dominant_spatial_map(rf: torch.Tensor) -> torch.Tensor:
    return rf if rf.ndim == 1 else rf[rf.square().sum(dim=1).argmax()]


def _centroid(rf: torch.Tensor, positions_degs: torch.Tensor) -> torch.Tensor:
    weights = rf.abs()
    return (weights[:, None] * positions_degs).sum(dim=0) / weights.sum()
