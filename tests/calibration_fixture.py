from __future__ import annotations

import numpy as np
import torch

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from models.response_snn import build_response_retina_model
from training.response_config import (
    ResponseDataConfig,
    ResponseEvaluationConfig,
    ResponseExperimentConfig,
    ResponseModelConfig,
    ResponseTrainingConfig,
)
from training.response_data import PreparedResponseData, ResponseSplit
from training.response_trainer import ResponseTrainer


def calibration_trainer() -> ResponseTrainer:
    data = calibration_data()
    config = calibration_config()
    priors = load_type_priors(
        "configs/rgc_type_priors.yaml",
        required_type_ids=("midget", "parasol"),
    )
    model = build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        macaque_photopic(
            dt_ms=5.0,
            cone_spacing_deg=0.1,
            eccentricity_deg=4.0,
        ),
        priors,
        support_radius_degs=0.2,
        readout_rate_tau_ms=50.0,
        surrogate_slope=5.0,
        parameter_sharing_mode="type_blind",
        enable_response_bias=False,
        enable_synaptic_gain=False,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
    )
    return ResponseTrainer(model, config, data, torch.device("cpu"))


def calibration_config() -> ResponseExperimentConfig:
    return ResponseExperimentConfig(
        seed=19,
        data=ResponseDataConfig("train", "validation", "test", 4),
        model=ResponseModelConfig(
            "configs/rgc_type_priors.yaml",
            0.2,
            50.0,
            5.0,
            parameter_sharing_mode="type_blind",
        ),
        training=ResponseTrainingConfig(1, 3, 1, 1, 1, 0.001, 1.0, 1),
        evaluation=ResponseEvaluationConfig(1, (0,), rf_finite_difference_checks=False),
    )


def calibration_data() -> PreparedResponseData:
    counts = torch.tensor(
        [[[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]],
    )
    split = ResponseSplit(
        cone_response=torch.tensor(
            [[[0.0, 0.0], [0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]]]
        ),
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=("source",),
        context_ids=("stationary",),
    )
    cells = CellMetadata(
        ids=("midget-on", "parasol-off"),
        type_ids=("midget", "parasol"),
        polarities=np.asarray([0, 1], dtype=np.int64),
        positions_degs=np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
        eccentricities_deg=np.asarray([4.0, 4.0], dtype=np.float32),
    )
    return PreparedResponseData(
        train=split,
        validation=split,
        test=split,
        cells=cells,
        cone_positions_degs=np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
        time_axis_seconds=np.arange(4, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(2, dtype=np.float32),
        normalization_std=np.ones(2, dtype=np.float32),
        fingerprint="calibration-fixture",
        input_identity=synthetic_input_identity(2, ("source",)),
    )
