from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any, Sequence

import torch

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc_types import RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder, TiedReadoutGeometry
from models.retina_snn import (
    RetinaModel,
    RetinaState,
    detach_state,
)
from training.bootstrap import (
    BootstrapContext,
    BootstrapReadouts,
    BootstrapRuntime,
    apply_crossfit_bootstrap,
)
from training.checkpointing import checkpoint_payload
from training.config import ExperimentConfig
from training.augmentation import AugmentedClip
from training.optimizer_step import run_optimizer_step
from training.schedule import objective_weights
from training.state import (
    BootstrapState,
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
        self.bootstrap_state = BootstrapState()
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
        target_region = (
            clean_target.float()
            if full_bptt
            else clean_target[:, training.burn_in_steps :].float()
        )
        persistent_prediction = self.decoder(history.rates, spatial_weights)
        weights = objective_weights(self.optimizer_step, self.config)
        prediction = persistent_prediction
        generator_auxiliary = persistent_prediction.new_zeros(())
        bootstrap_active = (
            self.model.training
            and torch.is_grad_enabled()
            and not full_bptt
            and self.optimizer_step
            < training.reconstruction_bootstrap_steps
        )
        if bootstrap_active:
            supervised = training.supervised_steps
            application = apply_crossfit_bootstrap(
                BootstrapReadouts(
                    rate_readout=history.rates[:, -supervised:],
                    generator_readout=history.generator_potential[
                        :, -supervised:
                    ],
                    target=target_region[:, -supervised:],
                    persistent_prediction=persistent_prediction[
                        :, -supervised:
                    ],
                ),
                BootstrapContext(
                    geometry=TiedReadoutGeometry(
                        spatial_weights=spatial_weights,
                        prior_gain=self.decoder.unit_gain.detach(),
                        gain_max=self.decoder.gain_max,
                    ),
                    reconstruction_scale=self.reconstruction_scale,
                    generator_auxiliary_scale=(
                        weights.generator_auxiliary_scale
                    ),
                ),
                BootstrapRuntime(
                    state=self.bootstrap_state,
                    parameters=tuple(self.model.parameters()),
                ),
            )
            prediction = torch.cat(
                (
                    persistent_prediction[:, :-supervised].detach(),
                    application.prediction,
                ),
                dim=1,
            )
            generator_auxiliary = application.generator_auxiliary
        losses = self.objective(
            prediction,
            target_region,
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
        if bootstrap_active:
            losses = replace(
                losses,
                total=losses.total + generator_auxiliary,
            )
        return losses, history, state

    def train_optimizer_step(
        self,
        clips: Sequence[AugmentedClip],
    ) -> OptimizerStepResult:
        return run_optimizer_step(self, clips)

    def checkpoint_payload(
        self,
        sampling_generator: torch.Generator,
        augmentation_generator: torch.Generator,
    ) -> dict[str, Any]:
        payload = checkpoint_payload(
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
        payload["bootstrap_state"] = asdict(self.bootstrap_state)
        return payload

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
        self.bootstrap_state = BootstrapState(**payload["bootstrap_state"])
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
