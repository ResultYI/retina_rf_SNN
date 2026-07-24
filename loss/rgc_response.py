from __future__ import annotations

from typing import assert_never

import torch
from torch.nn import functional as F

from data.rgc_response import ResponseTargetKind


class ResponseLossError(ValueError):
    pass


def response_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    if logits.shape != targets.shape or valid_mask.shape != targets.shape:
        raise ResponseLossError("logits, targets, and valid_mask must share a shape")
    if logits.ndim not in (3, 4):
        raise ResponseLossError(
            "response tensors must be [batch,time,cell] or "
            "[stimulus,trial,time,cell]"
        )
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ResponseLossError("response tensors must be finite")
    mask = valid_mask.to(dtype=logits.dtype)
    reduction_dims = tuple(range(logits.ndim - 1))
    denominator = mask.sum(dim=reduction_dims)
    if torch.any(denominator <= 0):
        raise ResponseLossError("Every cell needs at least one valid likelihood target")
    raw = response_nll_elements(logits, targets, target_kind)
    return ((raw * mask).sum(dim=reduction_dims) / denominator).mean()


def response_nll_elements(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
        case ResponseTargetKind.POISSON:
            expected_count = F.softplus(logits).clamp_min(1e-8)
            return F.poisson_nll_loss(
                expected_count,
                targets,
                log_input=False,
                full=False,
                reduction="none",
            )
        case _ as unreachable:
            assert_never(unreachable)


def expected_response(
    logits: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            return torch.sigmoid(logits)
        case ResponseTargetKind.POISSON:
            return F.softplus(logits)
        case _ as unreachable:
            assert_never(unreachable)


__all__ = [
    "ResponseLossError",
    "expected_response",
    "response_nll",
    "response_nll_elements",
]
