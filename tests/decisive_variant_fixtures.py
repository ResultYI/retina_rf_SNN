from __future__ import annotations

import numpy as np
import torch

from data.input_identity import legacy_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from training.response_config import (
    ParameterSharingMode,
    ResponseDataConfig,
    ResponseEvaluationConfig,
    ResponseExperimentConfig,
    ResponseModelConfig,
    ResponseTrainingConfig,
)
from training.response_data import PreparedResponseData, ResponseSplit


class StopAfterModelConstruction(RuntimeError):
    pass


class FakeVariantModel:
    def to(self, _device: torch.device) -> FakeVariantModel:
        return self


class FakeResponseTrainer:
    def __init__(self, *_args) -> None:
        self.optimizer = None
        self.sampling_generator = None


def response_config(
    mode: ParameterSharingMode,
    *,
    seed: int,
) -> ResponseExperimentConfig:
    return ResponseExperimentConfig(
        seed=seed,
        data=ResponseDataConfig(
            train_glob="train.h5",
            validation_glob="validation.h5",
            test_glob="test.h5",
            sequence_steps=4,
        ),
        model=ResponseModelConfig(
            type_prior_path="priors.yaml",
            support_radius_degs=0.2,
            readout_rate_tau_ms=10.0,
            surrogate_slope=3.0,
            parameter_sharing_mode=mode,
            enable_response_bias=True,
            enable_synaptic_gain=True,
        ),
        training=ResponseTrainingConfig(
            burn_in_steps=1,
            differentiable_steps=3,
            checkpoint_block_steps=1,
            batch_size=1,
            max_optimizer_steps=1,
            learning_rate=0.01,
            gradient_clip_norm=1.0,
            validation_interval_steps=1,
            stage0_calibration_enabled=True,
        ),
        evaluation=ResponseEvaluationConfig(
            rf_lag_steps=1,
            recovery_delays_ms=(0,),
        ),
    )


def prepared_response_data() -> PreparedResponseData:
    split = ResponseSplit(
        cone_response=torch.zeros(1, 4, 2),
        spike_counts=torch.zeros(1, 1, 4, 2),
        valid_mask=torch.ones(1, 1, 4, 2, dtype=torch.bool),
        source_ids=("source",),
        context_ids=("low",),
    )
    return PreparedResponseData(
        train=split,
        validation=split,
        test=split,
        cells=CellMetadata(
            ids=("c0", "c1"),
            type_ids=("midget", "parasol"),
            polarities=np.asarray([1, -1]),
            positions_degs=np.zeros((2, 2), dtype=np.float32),
            eccentricities_deg=np.ones(2, dtype=np.float32),
        ),
        cone_positions_degs=np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
        time_axis_seconds=np.asarray([0.0, 0.005, 0.010, 0.015], dtype=np.float32),
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(2, dtype=np.float32),
        normalization_std=np.ones(2, dtype=np.float32),
        fingerprint="fingerprint",
        input_identity=legacy_input_identity(),
    )
