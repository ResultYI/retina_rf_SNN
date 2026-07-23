from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from training.augmentation import AugmentedClip
from training.component_gradients import component_gradient_norms
from training.metrics import loss_metrics
from training.state import OptimizerStepResult

if TYPE_CHECKING:
    from training.trainer import RetinaTrainer


class OptimizerStepError(ValueError):
    pass


def run_optimizer_step(
    trainer: RetinaTrainer,
    clips: Sequence[AugmentedClip],
) -> OptimizerStepResult:
    expected = trainer.config.training.batch_size
    if len(clips) != expected:
        raise OptimizerStepError(
            "Optimizer step received the wrong batch size"
        )
    trainer.model.train()
    trainer.decoder.train()
    decoder_frozen = (
        trainer.optimizer_step
        < max(
            trainer.config.training.decoder_freeze_steps,
            trainer.config.training.reconstruction_bootstrap_steps,
        )
    )
    trainer.decoder.requires_grad_(not decoder_frozen)
    trainer.optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch = AugmentedClip.stack(clips)
    losses, history, _ = trainer.forward_clip(
        batch.noisy_input,
        batch.clean_target,
        checkpointed=True,
    )
    if not torch.isfinite(losses.total):
        raise OptimizerStepError("Training produced a non-finite loss")
    losses.total.backward()
    component_gradients = component_gradient_norms(
        trainer.model,
        trainer.decoder,
    )
    clipped_gradient_norm = float(
        clip_grad_norm_(
            (*trainer.model.parameters(), *trainer.decoder.parameters()),
            trainer.config.training.gradient_clip_norm,
        )
    )
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer_step += 1
    trainer.energy_state.observe(
        float(losses.energy.detach()),
        trainer.optimizer_step,
        trainer.config,
    )
    metrics = loss_metrics(losses, history)
    metrics.update(asdict(component_gradients))
    bootstrap_metrics = asdict(trainer.bootstrap_state)
    metrics.update(
        {
            f"bootstrap_{name}": (
                float(value) if value is not None else 0.0
            )
            for name, value in bootstrap_metrics.items()
        }
    )
    current_budget = trainer.energy_state.current_budget
    target_budget = trainer.energy_state.target_budget
    hard_energy = metrics["hard_energy"]
    metrics.update(
        {
            "current_budget": current_budget or 0.0,
            "target_budget": target_budget or 0.0,
            "current_energy_ratio": (
                hard_energy / current_budget
                if current_budget is not None
                else 0.0
            ),
            "target_energy_ratio": (
                hard_energy / target_budget
                if target_budget is not None
                else 0.0
            ),
            "energy_ema": trainer.energy_state.ema_energy or 0.0,
            "energy_dual": trainer.energy_state.dual,
            "lr_model": float(trainer.optimizer.param_groups[0]["lr"]),
            "lr_decoder": float(trainer.optimizer.param_groups[1]["lr"]),
            "decoder_frozen": float(decoder_frozen),
        }
    )
    return OptimizerStepResult(
        metrics=metrics,
        gradient_norm=clipped_gradient_norm,
        temporal_gradient_norm=component_gradients.temporal_gradient_norm,
        peak_memory_bytes=(
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0
        ),
    )


__all__ = ["OptimizerStepError", "run_optimizer_step"]
