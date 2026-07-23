from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from training.trainer import RetinaTrainer


def temporal_gradient_audit(
    trainer: RetinaTrainer,
    noisy_input: torch.Tensor,
    clean_target: torch.Tensor,
) -> dict[str, float | bool]:
    gradients: list[torch.Tensor] = []
    for full_bptt in (False, True):
        trainer.optimizer.zero_grad(set_to_none=True)
        losses, _, _ = trainer.forward_clip(
            noisy_input,
            clean_target,
            checkpointed=False,
            full_bptt=full_bptt,
        )
        losses.normalized_reconstruction.backward()
        pieces = [
            torch.zeros_like(parameter).flatten()
            if parameter.grad is None
            else parameter.grad.detach().flatten()
            for name, parameter in trainer.model.named_parameters()
            if any(token in name for token in ("tau", "gain", "mix"))
        ]
        gradients.append(torch.cat(pieces))
    truncated, full = gradients
    full_norm = float(full.norm())
    truncated_norm = float(truncated.norm())
    cosine = float(F.cosine_similarity(truncated, full, dim=0))
    ratio = truncated_norm / max(
        full_norm,
        torch.finfo(full.dtype).eps,
    )
    trainer.optimizer.zero_grad(set_to_none=True)
    return {
        "cosine": cosine,
        "norm_ratio": ratio,
        "truncated_norm": truncated_norm,
        "full_norm": full_norm,
        "passed": cosine >= 0.95 and 0.8 <= ratio <= 1.2,
    }


__all__ = ["temporal_gradient_audit"]
