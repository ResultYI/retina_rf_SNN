from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Sequence

import torch

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc_types import RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel, RetinaState
from training.checkpointing import checkpoint_payload
from training.config import ExperimentConfig
from training.augmentation import AugmentedClip
from training.forward_clip import ForwardClipRequest, forward_training_clip
from training.optimizer_step import run_optimizer_step
from training.state import (
    BootstrapState,
    EnergyBudgetState,
    OptimizerStepResult,
    RepresentationSelectionMetrics,
    ValidationState,
)


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
        return forward_training_clip(
            ForwardClipRequest(
                model=self.model,
                decoder=self.decoder,
                objective=self.objective,
                config=self.config,
                reconstruction_scale=self.reconstruction_scale,
                optimizer_step=self.optimizer_step,
                energy_state=self.energy_state,
                bootstrap_state=self.bootstrap_state,
                noisy_input=noisy_input,
                clean_target=clean_target,
                checkpointed=checkpointed,
                full_bptt=full_bptt,
            )
        )

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

    def record_representation(
        self,
        metrics: RepresentationSelectionMetrics,
    ) -> bool:
        return self.validation_state.observe_representation(metrics)


__all__ = [
    "EnergyBudgetState",
    "OptimizerStepResult",
    "RetinaTrainer",
    "TrainingError",
    "ValidationState",
    "temporal_gradient_audit",
]
