from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

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
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.0,
        )
        self.sampling_generator = torch.Generator().manual_seed(config.seed + 1)
        self.optimizer_step = 0
        self.best_nll = float("inf")
        self.baseline_rates = training_baseline_rates(
            data.train.spike_counts.flatten(0, 1),
            data.train.valid_mask.flatten(0, 1),
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
        supervised_counts = counts[:, self.config.training.burn_in_steps :]
        supervised_mask = mask[:, self.config.training.burn_in_steps :]
        likelihood = response_nll(
            output.spike_logits,
            supervised_counts,
            supervised_mask,
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
        free_running: bool = False,
    ) -> ResponseMetrics:
        self.model.eval()
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
                if free_running:
                    full_output, _ = self.model.forward_sequence(cones)
                    burn = self.config.training.burn_in_steps
                    output_logits = full_output.spike_logits[:, burn:]
                else:
                    history_counts = masked_history_counts(counts, mask)
                    output, _ = unroll_response(
                        ResponseUnrollRequest(
                            self.model,
                            cones,
                            history_counts,
                            self.config.training.burn_in_steps,
                            self.config.training.differentiable_steps,
                            self.config.training.checkpoint_block_steps,
                            False,
                        )
                    )
                    output_logits = output.spike_logits
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

    def save(self, path: str | Path) -> None:
        save_response_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            optimizer_step=self.optimizer_step,
            best_nll=self.best_nll,
            generator=self.sampling_generator,
            fingerprint=self.data.fingerprint,
            target_kind=self.data.target_kind.value,
            config=self.config,
        )


__all__ = ["ResponseStepResult", "ResponseTrainer"]
