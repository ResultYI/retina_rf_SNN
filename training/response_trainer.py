from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from evaluation.response_calibration import (
    LogitCalibrationRequest,
    fit_logit_calibration,
)
from evaluation.response_metrics import (
    ResponseMetrics,
    compute_response_metrics,
    training_baseline_rates,
)
from evaluation.response_predictions import (
    ConditionalHistoryMode,
    ResponseHistoryMode,
    ResponseHistoryModeError,
    ResponseHistoryTrial,
    ResponsePredictionRequest,
    collect_response_predictions,
    evaluation_history_counts,
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
from training.response_optimizer import (
    build_response_optimizer,
    configure_cell_residual_learning,
    freeze_threshold,
)
from training.response_readout_calibration import (
    Stage05ReadoutCalibrationRequest,
    Stage05ReadoutCalibrationResult,
    fit_stage05_readout_calibration,
)
from training.response_unroll import ResponseUnrollRequest, unroll_response


_configure_cell_residual_learning = configure_cell_residual_learning


@dataclass(frozen=True, slots=True)
class ResponseStepResult:
    loss: float
    likelihood: float
    physiology_prior: float
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class Stage0CalibrationResult:
    pre_train_nll: float
    post_train_nll: float
    fitted_bias: tuple[float, ...]


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
        configure_cell_residual_learning(
            model,
            learnable=config.training.learn_cell_residuals,
        )
        if config.training.freeze_threshold:
            freeze_threshold(model)
        burn_in = config.training.burn_in_steps
        self.baseline_rates = training_baseline_rates(
            data.train.spike_counts[:, :, burn_in:].flatten(0, 1),
            data.train.valid_mask[:, :, burn_in:].flatten(0, 1),
        ).to(device)
        self.stage0_result = None
        if config.training.stage0_calibration_enabled:
            self.stage0_result = self._run_stage0_calibration()
        self.stage05_result = None
        if config.training.stage05_readout_calibration_enabled:
            self.stage05_result = fit_stage05_readout_calibration(
                Stage05ReadoutCalibrationRequest(
                    model,
                    data.train,
                    data.target_kind,
                    burn_in,
                    device,
                )
            )
        self.optimizer = build_response_optimizer(model, config)
        self.sampling_generator = torch.Generator().manual_seed(config.seed + 1)
        self.optimizer_step = 0
        self.best_nll = float("inf")
        self.best_checkpoint_step = 0
        self.run_id = uuid4().hex
        self.parent_run_id: str | None = None

    def _run_stage0_calibration(self) -> Stage0CalibrationResult:
        train_predictions = collect_response_predictions(
            ResponsePredictionRequest(
                self.model,
                self.data.train,
                self.config.training.burn_in_steps,
                self.device,
                "observed",
            )
        )
        pre_train_nll = response_nll(
            train_predictions.logits,
            train_predictions.targets,
            train_predictions.valid_mask,
            self.data.target_kind,
        )
        result = fit_logit_calibration(
            LogitCalibrationRequest(
                train_predictions,
                train_predictions,
                self.data.target_kind,
                self.baseline_rates,
                "intercept",
                50,
            )
        )
        fitted_bias = torch.tensor(
            result.intercepts,
            device=self.model.rgc.response_bias.device,
            dtype=self.model.rgc.response_bias.dtype,
        )
        with torch.no_grad():
            self.model.rgc.response_bias.copy_(fitted_bias)
        return Stage0CalibrationResult(
            pre_train_nll=float(pre_train_nll.detach()),
            post_train_nll=result.train_metrics.nll,
            fitted_bias=result.intercepts,
        )

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
        predictions = collect_response_predictions(
            ResponsePredictionRequest(
                evaluation_model,
                split,
                self.config.training.burn_in_steps,
                self.device,
                history_mode,
            )
        )
        return compute_response_metrics(
            predictions.logits,
            predictions.targets,
            predictions.valid_mask,
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


__all__ = [
    "ConditionalHistoryMode",
    "ResponseHistoryMode",
    "ResponseHistoryModeError",
    "ResponseHistoryTrial",
    "ResponseStepResult",
    "Stage0CalibrationResult",
    "Stage05ReadoutCalibrationResult",
    "ResponseTrainer",
    "evaluation_history_counts",
]
