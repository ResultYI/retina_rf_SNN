from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class MechanisticLossError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def expected_bernoulli_nll(
    logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != teacher_probability.shape or valid_mask.shape != logits.shape:
        raise MechanisticLossError("likelihood tensors must share one shape")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(teacher_probability).all()):
        raise MechanisticLossError("likelihood tensors must be finite")
    mask = valid_mask.to(dtype=logits.dtype)
    denominator = mask.sum().clamp_min(1)
    return ((F.softplus(logits) - teacher_probability * logits) * mask).sum() / denominator


__all__ = ["MechanisticLossError", "expected_bernoulli_nll"]
