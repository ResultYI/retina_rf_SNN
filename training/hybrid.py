from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch
from torch.nn.utils import clip_grad_norm_

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.decoder.local_decoder import (
    LocalDecoder,
    LocalDecoderDiagnostics,
    LocalDecoderOutput,
)
from models.retina_snn import (
    RetinaSNNCore,
    RetinaSNNState,
    RetinaStepDiagnostics,
    detach_state,
)


class HybridTrainingError(ValueError):
    pass


class TrainingStage(StrEnum):
    DECODER_WARMUP = "decoder_warmup"
    CORE_FINETUNE = "core_finetune"


@dataclass(frozen=True, slots=True)
class RetinaTargets:
    fine: torch.Tensor
    coarse: torch.Tensor


@dataclass(frozen=True, slots=True)
class RetinaTrainingBatch:
    x_cone: torch.Tensor
    targets: RetinaTargets


@dataclass(frozen=True, slots=True)
class HybridTrainingConfig:
    t_bptt: int = 8
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.t_bptt < 1:
            raise HybridTrainingError("t_bptt must be positive")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise HybridTrainingError("grad_clip_norm must be positive when set")


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    losses: RetinaLosses
    state: RetinaSNNState
    core_diagnostics: RetinaStepDiagnostics
    decoder_diagnostics: LocalDecoderDiagnostics


class HybridRetinaTrainer:
    def __init__(
        self,
        core: RetinaSNNCore,
        decoder: LocalDecoder,
        objective: RetinaObjective,
        optimizer: torch.optim.Optimizer,
        config: HybridTrainingConfig,
    ) -> None:
        self.core = core
        self.decoder = decoder
        self.objective = objective
        self.optimizer = optimizer
        self.config = config

    def train_batch(
        self,
        batch: RetinaTrainingBatch,
        stage: TrainingStage,
        state: RetinaSNNState | None = None,
    ) -> TrainingStepResult:
        x_cone = batch.x_cone
        if x_cone.ndim != 3 or x_cone.shape[1] < 1:
            raise HybridTrainingError("x_cone must have shape [batch,time,Ncone]")
        if state is None:
            state = self.core.initial_state(
                x_cone.shape[0],
                x_cone.device,
                x_cone.dtype,
            )

        split = max(0, x_cone.shape[1] - self.config.t_bptt)
        if split:
            with torch.no_grad():
                _, state = self.core.forward_sequence(x_cone[:, :split], state)
            state = detach_state(state)
        train_window = x_cone[:, split:]

        self.optimizer.zero_grad(set_to_none=True)
        match stage:
            case TrainingStage.DECODER_WARMUP:
                with torch.no_grad():
                    rgc_history, state, diagnostics = self.core.forward_sequence(
                        train_window,
                        state,
                        return_diagnostics=True,
                    )
            case TrainingStage.CORE_FINETUNE:
                rgc_history, state, diagnostics = self.core.forward_sequence(
                    train_window,
                    state,
                    return_diagnostics=True,
                )
            case unreachable:
                assert_never(unreachable)

        final_rgc = _last_rgc_output(rgc_history)
        prediction, decoder_diagnostics = self.decoder(
            final_rgc,
            return_diagnostics=True,
        )
        losses = self.objective(
            prediction,
            batch.targets,
            rgc_history,
            self.decoder.residual_weight_penalty(),
        )
        losses.total.backward()
        if self.config.grad_clip_norm is not None:
            clip_grad_norm_(
                (*self.core.parameters(), *self.decoder.parameters()),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        return TrainingStepResult(
            losses=losses.detached(),
            state=detach_state(state),
            core_diagnostics=diagnostics[-1],
            decoder_diagnostics=decoder_diagnostics,
        )


def _last_rgc_output(history: RGCOutput) -> RGCOutput:
    return RGCOutput(
        spikes=_last_populations(history.spikes),
        rates=_last_populations(history.rates),
    )


def _last_populations(
    populations: RGCPopulationTensors,
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        midget=populations.midget[:, -1],
        parasol=populations.parasol[:, -1],
        residual=populations.residual[:, -1],
    )
