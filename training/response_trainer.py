from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, assert_never
from uuid import uuid4

import torch
from torch import nn

from evaluation.response_metrics import (
    ResponseMetrics,
    compute_response_metrics,
    training_baseline_rates,
)
from loss.rgc_response import response_nll
from models.response_snn import ResponseRetinaModel
from training.response_checkpointing import save_response_checkpoint
from training.response_config import ResponseExperimentConfig
from training.response_data import (
    PreparedResponseData,
    ResponseSplit,
    masked_history_counts,
)
from training.response_unroll import ResponseUnrollRequest, unroll_response


ResponseHistoryMode: TypeAlias = Literal[
    "observed",
    "zero",
    "shuffled",
    "free_running",
]
ConditionalHistoryMode: TypeAlias = Literal["observed", "zero", "shuffled"]


class ResponseHistoryModeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseHistoryTrial:
    split: ResponseSplit
    stimulus_index: int
    trial_index: int


@dataclass(frozen=True, slots=True)
class ResponseStepResult:
    loss: float
    likelihood: float
    physiology_prior: float
    gradient_norm: float


class ResponseTrainer:
    def __init__(
        self,
        model: ResponseRetinaModel,
        config: ResponseExperimentConfig,
        data: PreparedResponseData,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.data = data
        self.device = device
        _configure_cell_residual_learning(
            model,
            learnable=config.training.learn_cell_residuals,
        )
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.training.learning_rate,
            weight_decay=0.0,
        )
        self.sampling_generator = torch.Generator().manual_seed(config.seed + 1)
        self.optimizer_step = 0
        self.best_nll = float("inf")
        self.best_checkpoint_step = 0
        self.run_id = uuid4().hex
        self.parent_run_id: str | None = None
        burn_in = config.training.burn_in_steps
        self.baseline_rates = training_baseline_rates(
            data.train.spike_counts[:, :, burn_in:].flatten(0, 1),
            data.train.valid_mask[:, :, burn_in:].flatten(0, 1),
        ).to(device)

    def train_step(
        self,
        cones: torch.Tensor,
        counts: torch.Tensor,
        mask: torch.Tensor,
    ) -> ResponseStepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        history_counts = masked_history_counts(counts, mask)
        output, _ = unroll_response(
            ResponseUnrollRequest(
                model=self.model,
                cone_response=cones,
                observed_counts=history_counts,
                burn_in_steps=self.config.training.burn_in_steps,
                differentiable_steps=self.config.training.differentiable_steps,
                checkpoint_block_steps=self.config.training.checkpoint_block_steps,
                checkpointed=True,
            )
        )
        supervision = self.config.training.supervision_slice
        likelihood = response_nll(
            output.spike_logits[:, supervision],
            counts[:, self.config.training.burn_in_steps :][:, supervision],
            mask[:, self.config.training.burn_in_steps :][:, supervision],
            self.data.target_kind,
        )
        physiology_prior = self.model.rgc.physiology_prior_penalty()
        loss = likelihood + physiology_prior
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip_norm,
        )
        self.optimizer.step()
        self.optimizer_step += 1
        return ResponseStepResult(
            loss=float(loss.detach()),
            likelihood=float(likelihood.detach()),
            physiology_prior=float(physiology_prior.detach()),
            gradient_norm=float(gradient.detach()),
        )

    @torch.no_grad()
    def evaluate(
        self,
        split: ResponseSplit,
        *,
        history_mode: ResponseHistoryMode = "observed",
        model: ResponseRetinaModel | None = None,
    ) -> ResponseMetrics:
        evaluation_model = self.model if model is None else model
        evaluation_model.eval()
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for stimulus in range(split.cone_response.shape[0]):
            stimulus_logits: list[torch.Tensor] = []
            stimulus_targets: list[torch.Tensor] = []
            stimulus_masks: list[torch.Tensor] = []
            cones = split.cone_response[stimulus : stimulus + 1].to(self.device)
            for trial in range(split.spike_counts.shape[1]):
                counts = split.spike_counts[
                    stimulus : stimulus + 1, trial
                ].to(self.device)
                mask = split.valid_mask[stimulus : stimulus + 1, trial].to(
                    self.device
                )
                match history_mode:
                    case "free_running":
                        full_output, _ = evaluation_model.forward_sequence(cones)
                        burn = self.config.training.burn_in_steps
                        output_logits = full_output.spike_logits[:, burn:]
                    case "observed" | "zero" | "shuffled" as conditional_mode:
                        history_counts = evaluation_history_counts(
                            ResponseHistoryTrial(split, stimulus, trial),
                            conditional_mode,
                        ).to(self.device)
                        output, _ = unroll_response(
                            ResponseUnrollRequest(
                                evaluation_model,
                                cones,
                                history_counts,
                                self.config.training.burn_in_steps,
                                self.config.training.differentiable_steps,
                                self.config.training.checkpoint_block_steps,
                                False,
                            )
                        )
                        output_logits = output.spike_logits
                    case unreachable:
                        assert_never(unreachable)
                stimulus_logits.append(output_logits.squeeze(0))
                stimulus_targets.append(
                    counts[:, self.config.training.burn_in_steps :].squeeze(0)
                )
                stimulus_masks.append(
                    mask[:, self.config.training.burn_in_steps :].squeeze(0)
                )
            logits.append(torch.stack(stimulus_logits))
            targets.append(torch.stack(stimulus_targets))
            masks.append(torch.stack(stimulus_masks))
        return compute_response_metrics(
            torch.stack(logits),
            torch.stack(targets),
            torch.stack(masks),
            self.data.target_kind,
            self.baseline_rates,
        )

    def save(self, path: str | Path, checkpoint_kind: str) -> None:
        save_response_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            optimizer_step=self.optimizer_step,
            best_nll=self.best_nll,
            best_checkpoint_step=self.best_checkpoint_step,
            generator=self.sampling_generator,
            fingerprint=self.data.fingerprint,
            target_kind=self.data.target_kind.value,
            config=self.config,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            checkpoint_kind=checkpoint_kind,
        )


def evaluation_history_counts(
    trial: ResponseHistoryTrial,
    history_mode: ConditionalHistoryMode,
) -> torch.Tensor:
    match history_mode:
        case "observed":
            history_trial = trial.trial_index
        case "zero":
            return torch.zeros_like(
                trial.split.spike_counts[
                    trial.stimulus_index : trial.stimulus_index + 1,
                    trial.trial_index,
                ]
            )
        case "shuffled":
            trial_count = trial.split.spike_counts.shape[1]
            if trial_count < 2:
                raise ResponseHistoryModeError(
                    "Shuffled response history requires at least two trials"
                )
            history_trial = (trial.trial_index + 1) % trial_count
        case unreachable:
            assert_never(unreachable)
    counts = trial.split.spike_counts[
        trial.stimulus_index : trial.stimulus_index + 1,
        history_trial,
    ]
    mask = trial.split.valid_mask[
        trial.stimulus_index : trial.stimulus_index + 1,
        history_trial,
    ]
    return masked_history_counts(counts, mask)


def _configure_cell_residual_learning(
    model: nn.Module,
    *,
    learnable: bool,
) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("rgc.") and name.endswith(".cell_residual_raw"):
            parameter.requires_grad_(learnable)


__all__ = [
    "ConditionalHistoryMode",
    "ResponseHistoryMode",
    "ResponseHistoryModeError",
    "ResponseHistoryTrial",
    "ResponseStepResult",
    "ResponseTrainer",
    "evaluation_history_counts",
]
