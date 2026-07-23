from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc_types import RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import (
    RetinaModel,
    RetinaState,
    detach_state,
)
from training.checkpointing import checkpoint_payload
from training.config import ExperimentConfig
from training.gradient_audit import temporal_gradient_audit
from training.metrics import gradient_norm, loss_metrics
from training.augmentation import AugmentedClip
from training.schedule import objective_weights
from training.state import (
    EnergyBudgetState,
    OptimizerStepResult,
    ValidationState,
)
from training.unroll import ForwardRegionRequest, forward_region


class TrainingError(ValueError):
    pass


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
        self.validation_state = ValidationState()
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
            history, state = forward_region(
                ForwardRegionRequest(
                    model=self.model,
                    region=region,
                    state=state,
                    spatial_weights=spatial_weights,
                    checkpointed=checkpointed,
                    block_steps=training.checkpoint_block_steps,
                )
            )
        prediction = self.decoder(history.rates, spatial_weights)
        weights = objective_weights(self.optimizer_step, self.config)
        losses = self.objective(
            prediction,
            clean_target.float() if full_bptt else clean_target[:, training.burn_in_steps :].float(),
            history,
            self.model.rgc,
            spatial_weights,
            reconstruction_scale=self.reconstruction_scale,
            energy_budget=self.energy_state.current_budget,
            energy_dual=self.energy_state.dual,
            energy_weight=weights.energy,
            wiring_weight=weights.wiring,
            variance_weight=weights.variance,
            phenotype_repulsion_weight=weights.phenotype_repulsion,
            homeostasis_weight=weights.homeostasis,
            supervised_steps=training.supervised_steps,
        )
        return losses, history, state

    def train_optimizer_step(
        self,
        clips: Sequence[AugmentedClip],
    ) -> OptimizerStepResult:
        expected = self.config.training.batch_size
        if len(clips) != expected:
            raise TrainingError("Optimizer step received the wrong batch size")
        self.model.train()
        self.decoder.train()
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        batch = AugmentedClip.stack(clips)
        losses, history, _ = self.forward_clip(
            batch.noisy_input,
            batch.clean_target,
            checkpointed=True,
        )
        if not torch.isfinite(losses.total):
            raise TrainingError("Training produced a non-finite loss")
        losses.total.backward()
        temporal_gradient_norm = gradient_norm(
            parameter
            for name, parameter in self.model.named_parameters()
            if any(token in name for token in ("tau", "gain", "mix"))
        )
        clipped_gradient_norm = float(
            clip_grad_norm_(
                (*self.model.parameters(), *self.decoder.parameters()),
                self.config.training.gradient_clip_norm,
            )
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer_step += 1
        self.energy_state.observe(
            float(losses.energy.detach()),
            self.optimizer_step,
            self.config,
        )
        metrics = loss_metrics(losses, history)
        current_budget = self.energy_state.current_budget
        target_budget = self.energy_state.target_budget
        hard_energy = metrics["hard_energy"]
        metrics.update(
            {
                "current_budget": current_budget or 0.0,
                "target_budget": target_budget or 0.0,
                "current_energy_ratio": (
                    hard_energy / current_budget if current_budget is not None else 0.0
                ),
                "target_energy_ratio": (
                    hard_energy / target_budget if target_budget is not None else 0.0
                ),
                "energy_ema": self.energy_state.ema_energy or 0.0,
                "energy_dual": self.energy_state.dual,
                "lr_model": float(self.optimizer.param_groups[0]["lr"]),
                "lr_decoder": float(self.optimizer.param_groups[1]["lr"]),
            }
        )
        return OptimizerStepResult(
            metrics=metrics,
            gradient_norm=clipped_gradient_norm,
            temporal_gradient_norm=temporal_gradient_norm,
            peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        )

    def checkpoint_payload(
        self,
        sampling_generator: torch.Generator,
        augmentation_generator: torch.Generator,
    ) -> dict[str, Any]:
        return checkpoint_payload(
            optimizer_step=self.optimizer_step,
            model=self.model,
            decoder=self.decoder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            energy_state=self.energy_state,
            validation_state=self.validation_state,
            sampling_generator=sampling_generator,
            augmentation_generator=augmentation_generator,
            config=self.config,
        )

    def restore(
        self,
        payload: dict[str, Any],
        sampling_generator: torch.Generator,
        augmentation_generator: torch.Generator,
    ) -> None:
        self.model.load_state_dict(payload["model"])
        self.decoder.load_state_dict(payload["decoder"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.optimizer_step = int(payload["optimizer_step"])
        self.energy_state = EnergyBudgetState(**payload["energy_state"])
        self.validation_state = ValidationState(**payload["validation_state"])
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        sampling_generator.set_state(rng["sampling"].cpu())
        augmentation_generator.set_state(rng["augmentation"].cpu())

    def record_validation(
        self,
        optimizer_step: int,
        reconstruction_mse: float,
        target_energy_ratio: float | None,
    ) -> tuple[bool, bool]:
        return self.validation_state.observe(
            optimizer_step,
            reconstruction_mse,
            target_energy_ratio,
            self.config,
        )


__all__ = [
    "EnergyBudgetState",
    "OptimizerStepResult",
    "RetinaTrainer",
    "TrainingError",
    "ValidationState",
    "temporal_gradient_audit",
]
