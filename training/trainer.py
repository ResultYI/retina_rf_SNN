from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc_types import RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import (
    RetinaModel,
    RetinaState,
    detach_state,
    state_from_tensors,
    state_to_tensors,
)
from training.checkpointing import checkpoint_payload
from training.config import ExperimentConfig
from training.data import AugmentedClip


class TrainingError(ValueError):
    pass


@dataclass(slots=True)
class EnergyBudgetState:
    reference_energy: float | None = None
    ema_energy: float | None = None
    budget: float | None = None
    dual: float = 0.0

    def observe(self, energy: float, optimizer_step: int, config: ExperimentConfig) -> None:
        objective = config.objective
        training = config.training
        self.ema_energy = energy if self.ema_energy is None else 0.95 * self.ema_energy + 0.05 * energy
        if optimizer_step <= training.reconstruction_bootstrap_steps:
            self.reference_energy = self.ema_energy
            self.budget = None
            self.dual = 0.0
            return
        if self.reference_energy is None:
            self.reference_energy = self.ema_energy
        ramp_width = max(
            1,
            training.budget_ramp_end_step - training.reconstruction_bootstrap_steps,
        )
        ramp = min(
            1.0,
            (optimizer_step - training.reconstruction_bootstrap_steps) / ramp_width,
        )
        ratio = 1.0 + ramp * (objective.energy_budget_ratio - 1.0)
        self.budget = max(self.reference_energy * ratio, 1e-12)
        violation = max(0.0, self.ema_energy / self.budget - 1.0)
        self.dual = min(objective.dual_max, max(0.0, self.dual + objective.dual_lr * violation))


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    metrics: dict[str, float]
    gradient_norm: float
    temporal_gradient_norm: float
    peak_memory_bytes: int


class RetinaTrainer:
    def __init__(
        self,
        model: RetinaModel,
        decoder: TiedLocalDecoder,
        objective: RetinaObjective,
        config: ExperimentConfig,
        reconstruction_scale: float,
    ) -> None:
        if not math.isfinite(reconstruction_scale) or reconstruction_scale <= 0:
            raise TrainingError("reconstruction_scale must be positive and finite")
        self.model = model
        self.decoder = decoder
        self.objective = objective
        self.config = config
        self.reconstruction_scale = reconstruction_scale
        self.optimizer = torch.optim.AdamW(
            (
                {"name": "model", "params": model.parameters(), "lr": config.training.core_lr},
                {"name": "decoder", "params": decoder.parameters(), "lr": config.training.decoder_lr},
            ),
            weight_decay=0.0,
        )

        def lr_multiplier(step: int) -> float:
            progress = min(step / max(1, config.training.max_optimizer_steps), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_multiplier
        )
        self.energy_state = EnergyBudgetState()
        self.optimizer_step = 0

    def forward_clip(
        self,
        noisy_input: torch.Tensor,
        clean_target: torch.Tensor,
        *,
        checkpointed: bool,
        full_bptt: bool = False,
    ) -> tuple[RetinaLosses, RGCOutput, RetinaState]:
        if noisy_input.shape != clean_target.shape or noisy_input.ndim != 3:
            raise TrainingError("Input and target must match [batch,time,cone]")
        training = self.config.training
        if noisy_input.shape[1] != self.config.data.sequence_steps:
            raise TrainingError("Clip length does not match data.sequence_steps")
        state = self.model.initial_state(
            noisy_input.shape[0], noisy_input.device, torch.float32
        )
        spatial_weights = self.model.rgc.compute_spatial_weights()
        if full_bptt:
            history, state = self.model.forward_sequence(
                noisy_input.float(), state, spatial_weights=spatial_weights
            )
        else:
            with torch.no_grad():
                _, state = self.model.forward_sequence(
                    noisy_input[:, : training.burn_in_steps].float(),
                    state,
                    spatial_weights=spatial_weights,
                )
            state = detach_state(state)
            region = noisy_input[:, training.burn_in_steps :].float()
            if region.shape[1] != training.differentiable_steps:
                raise TrainingError("Differentiable region length is inconsistent")
            if training.context_only_steps + training.supervised_steps != region.shape[1]:
                raise TrainingError("Context and supervised regions are inconsistent")
            history, state = self._forward_region(
                region, state, spatial_weights, checkpointed
            )
        prediction = self.decoder(history.rates, spatial_weights)
        penalty_scale = min(
            1.0,
            self.optimizer_step / max(1, training.reconstruction_bootstrap_steps),
        )
        losses = self.objective(
            prediction,
            clean_target.float() if full_bptt else clean_target[:, training.burn_in_steps :].float(),
            history,
            self.model.rgc,
            spatial_weights,
            reconstruction_scale=self.reconstruction_scale,
            energy_budget=self.energy_state.budget,
            energy_dual=self.energy_state.dual,
            energy_weight=penalty_scale,
            wiring_weight=penalty_scale * self.config.objective.wiring_target_gradient_ratio,
            diversity_weight=penalty_scale * self.config.objective.diversity_target_gradient_ratio,
            supervised_steps=training.supervised_steps,
        )
        return losses, history, state

    def _forward_region(
        self,
        region: torch.Tensor,
        state: RetinaState,
        spatial_weights: torch.Tensor,
        checkpointed: bool,
    ) -> tuple[RGCOutput, RetinaState]:
        if not checkpointed:
            return self.model.forward_sequence(
                region, state, spatial_weights=spatial_weights
            )
        flat_state = state_to_tensors(state)
        histories: list[list[torch.Tensor]] = [[], [], [], []]
        block_steps = self.config.training.checkpoint_block_steps
        for start in range(0, region.shape[1], block_steps):
            block = region[:, start : start + block_steps]

            def run_block(
                block_input: torch.Tensor,
                cached_weights: torch.Tensor,
                *state_values: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                output, next_state = self.model.forward_sequence(
                    block_input,
                    state_from_tensors(tuple(state_values)),
                    spatial_weights=cached_weights,
                )
                return (
                    *state_to_tensors(next_state),
                    output.hard_spikes,
                    output.spike_probability,
                    output.rates,
                    output.generator_potential,
                )

            values = checkpoint(
                run_block,
                block,
                spatial_weights,
                *flat_state,
                use_reentrant=False,
            )
            flat_state = tuple(values[:8])
            for target, value in zip(histories, values[8:], strict=True):
                target.append(value)
        output = RGCOutput(*(torch.cat(values, dim=1) for values in histories))
        return output, state_from_tensors(flat_state)

    def train_optimizer_step(
        self,
        clips: Sequence[AugmentedClip],
    ) -> OptimizerStepResult:
        expected = self.config.training.gradient_accumulation_steps
        if len(clips) != expected:
            raise TrainingError("Optimizer step received the wrong accumulation count")
        self.model.train()
        self.decoder.train()
        self.optimizer.zero_grad(set_to_none=True)
        rows: list[dict[str, float]] = []
        energies: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for clip in clips:
            losses, history, _ = self.forward_clip(
                clip.noisy_input,
                clip.clean_target,
                checkpointed=True,
            )
            if not torch.isfinite(losses.total):
                raise TrainingError("Training produced a non-finite loss")
            (losses.total / expected).backward()
            energies.append(float(losses.energy.detach()))
            rows.append(_loss_metrics(losses, history))
        temporal_gradient_norm = _gradient_norm(
            parameter
            for name, parameter in self.model.named_parameters()
            if any(token in name for token in ("tau", "gain", "mix"))
        )
        gradient_norm = float(
            clip_grad_norm_(
                (*self.model.parameters(), *self.decoder.parameters()),
                self.config.training.gradient_clip_norm,
            )
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer_step += 1
        self.energy_state.observe(
            float(np.mean(energies)), self.optimizer_step, self.config
        )
        metrics = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        metrics.update(
            {
                "energy_budget": self.energy_state.budget or 0.0,
                "energy_ema": self.energy_state.ema_energy or 0.0,
                "energy_dual": self.energy_state.dual,
                "lr_model": float(self.optimizer.param_groups[0]["lr"]),
                "lr_decoder": float(self.optimizer.param_groups[1]["lr"]),
            }
        )
        return OptimizerStepResult(
            metrics=metrics,
            gradient_norm=gradient_norm,
            temporal_gradient_norm=temporal_gradient_norm,
            peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        )

    def checkpoint_payload(
        self, augmentation_generator: torch.Generator
    ) -> dict[str, Any]:
        return checkpoint_payload(
            optimizer_step=self.optimizer_step,
            model=self.model,
            decoder=self.decoder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            energy_state=self.energy_state,
            augmentation_generator=augmentation_generator,
            config=self.config,
        )

    def restore(
        self,
        payload: dict[str, Any],
        augmentation_generator: torch.Generator,
    ) -> None:
        self.model.load_state_dict(payload["model"])
        self.decoder.load_state_dict(payload["decoder"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.optimizer_step = int(payload["optimizer_step"])
        self.energy_state = EnergyBudgetState(**payload["energy_state"])
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        augmentation_generator.set_state(rng["augmentation"].cpu())


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
    ratio = truncated_norm / max(full_norm, torch.finfo(full.dtype).eps)
    trainer.optimizer.zero_grad(set_to_none=True)
    return {
        "cosine": cosine,
        "norm_ratio": ratio,
        "truncated_norm": truncated_norm,
        "full_norm": full_norm,
        "passed": cosine >= 0.95 and 0.8 <= ratio <= 1.2,
    }


def _loss_metrics(losses: RetinaLosses, output: RGCOutput) -> dict[str, float]:
    return {
        "loss_total": float(losses.total.detach()),
        "reconstruction": float(losses.reconstruction.detach()),
        "normalized_reconstruction": float(losses.normalized_reconstruction.detach()),
        "energy": float(losses.energy.detach()),
        "energy_penalty": float(losses.energy_penalty.detach()),
        "energy_violation": float(losses.energy_violation.detach()),
        "wiring": float(losses.wiring.detach()),
        "variance_floor": float(losses.variance_floor.detach()),
        "phenotype_repulsion": float(losses.phenotype_repulsion.detach()),
        "homeostasis": float(losses.homeostasis.detach()),
        "mean_rate": float(output.rates.mean().detach()),
        "active_unit_fraction": float((output.rates.amax(dim=1) > 0).float().mean().detach()),
    }


def _gradient_norm(parameters: Any) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


__all__ = [
    "EnergyBudgetState",
    "OptimizerStepResult",
    "RetinaTrainer",
    "TrainingError",
    "temporal_gradient_audit",
]
