from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from models.response_snn import build_response_retina_model
from training.response_config import (
    ResponseConfigurationError,
    ResponseDataConfig,
    ResponseEvaluationConfig,
    ResponseExperimentConfig,
    ResponseModelConfig,
    ResponseTrainingConfig,
)
from training.response_data import PreparedResponseData, ResponseSplit
from training.response_trainer import ResponseTrainer
from training.response_readout_calibration import (
    Stage05ReadoutCalibrationRequest,
    _trace_readout_features,
)


def test_stage05_requires_stage0_bias_and_direct_readout() -> None:
    # Given
    config = _config(stage05=True)

    # When / Then
    for model in (
        replace(config.model, enable_response_bias=False),
        replace(config.model, enable_direct_readout=False),
    ):
        with pytest.raises(ResponseConfigurationError, match="Stage0.5"):
            replace(config, model=model)
    with pytest.raises(ResponseConfigurationError, match="Stage0"):
        replace(
            config,
            training=replace(config.training, stage0_calibration_enabled=False),
        )


def test_stage05_is_train_only_and_changes_only_existing_readout() -> None:
    # Given
    data = _data()
    altered = replace(
        data,
        validation=_altered_split(data.validation),
        test=_altered_split(data.test),
    )
    stage0_only = _trainer(_config(stage05=False), data)

    # When
    calibrated = _trainer(_config(stage05=True), data)
    calibrated_with_altered_holdouts = _trainer(_config(stage05=True), altered)

    # Then
    assert calibrated.stage05_result is not None
    assert calibrated.stage05_result.post_train_nll <= (
        calibrated.stage05_result.pre_train_nll + 1e-6
    )
    assert calibrated.optimizer_step == 0
    assert torch.count_nonzero(calibrated.model.rgc.bipolar_readout_gain) > 0
    assert torch.count_nonzero(calibrated.model.rgc.amacrine_readout_gain) > 0
    for name, parameter in calibrated.model.named_parameters():
        reference = dict(stage0_only.model.named_parameters())[name]
        if name not in {
            "rgc.response_bias",
            "rgc.bipolar_readout_gain",
            "rgc.amacrine_readout_gain",
        }:
            assert torch.equal(parameter, reference), name
    for name in (
        "response_bias",
        "bipolar_readout_gain",
        "amacrine_readout_gain",
    ):
        assert torch.equal(
            getattr(calibrated.model.rgc, name),
            getattr(calibrated_with_altered_holdouts.model.rgc, name),
        )


def test_disabled_stage05_is_an_exact_value_noop() -> None:
    # Given
    data = _data()
    model = _model(data)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # When
    trainer = ResponseTrainer(model, _config(stage05=False, stage0=False), data, torch.device("cpu"))

    # Then
    assert trainer.stage05_result is None
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_stage05_batches_trials_for_each_stimulus() -> None:
    data = _data()
    repeated = replace(
        data.train,
        spike_counts=data.train.spike_counts.expand(-1, 3, -1, -1).clone(),
        valid_mask=data.train.valid_mask.expand(-1, 3, -1, -1).clone(),
    )
    model = _model(data)
    request = Stage05ReadoutCalibrationRequest(
        model,
        repeated,
        data.target_kind,
        1,
        torch.device("cpu"),
    )

    with patch.object(model, "step", wraps=model.step) as step:
        traces = _trace_readout_features(request)
    sequential = []
    for trial in range(repeated.spike_counts.shape[1]):
        trial_split = replace(
            repeated,
            spike_counts=repeated.spike_counts[:, trial : trial + 1],
            valid_mask=repeated.valid_mask[:, trial : trial + 1],
        )
        sequential.append(
            _trace_readout_features(replace(request, train=trial_split))
        )

    assert step.call_count == repeated.cone_response.shape[0] * repeated.cone_response.shape[1]
    assert traces.targets.shape == repeated.spike_counts[:, :, 1:].shape
    assert torch.allclose(
        traces.features,
        torch.cat([value.features for value in sequential], dim=1),
    )
    assert torch.allclose(
        traces.base_logits,
        torch.cat([value.base_logits for value in sequential], dim=1),
    )
    assert torch.equal(
        traces.targets,
        torch.cat([value.targets for value in sequential], dim=1),
    )


def test_stage05_can_use_conditional_probabilities_without_changing_history() -> None:
    # Given
    data = _data()
    model = _model(data)
    expected = torch.full_like(data.train.spike_counts, 0.25)
    request = Stage05ReadoutCalibrationRequest(
        model,
        data.train,
        data.target_kind,
        1,
        torch.device("cpu"),
        expected,
    )

    # When
    traces = _trace_readout_features(request)

    # Then
    torch.testing.assert_close(traces.targets, expected[:, :, 1:])


def _trainer(
    config: ResponseExperimentConfig,
    data: PreparedResponseData,
) -> ResponseTrainer:
    return ResponseTrainer(_model(data), config, data, torch.device("cpu"))


def _model(data: PreparedResponseData):
    priors = load_type_priors(
        "configs/rgc_type_priors.yaml",
        required_type_ids=("midget", "parasol"),
    )
    return build_response_retina_model(
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
        enable_response_bias=True,
        enable_synaptic_gain=True,
        enable_direct_readout=True,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
    )


def _config(
    *,
    stage05: bool,
    stage0: bool = True,
) -> ResponseExperimentConfig:
    return ResponseExperimentConfig(
        seed=19,
        data=ResponseDataConfig("train", "validation", "test", 6),
        model=ResponseModelConfig(
            "configs/rgc_type_priors.yaml",
            0.2,
            50.0,
            5.0,
            parameter_sharing_mode="type_blind",
            enable_response_bias=True,
            enable_synaptic_gain=True,
            enable_direct_readout=True,
        ),
        training=ResponseTrainingConfig(
            1,
            5,
            1,
            1,
            1,
            0.001,
            1.0,
            1,
            stage0_calibration_enabled=stage0,
            stage05_readout_calibration_enabled=stage05,
            freeze_threshold=True,
        ),
        evaluation=ResponseEvaluationConfig(
            1,
            (0,),
            rf_finite_difference_checks=False,
        ),
    )


def _data() -> PreparedResponseData:
    cones = torch.tensor(
        [
            [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.5], [1.0, -0.5], [-1.0, 1.0], [1.0, -1.0]],
            [[0.0, -1.0], [0.0, 1.0], [0.5, -1.0], [-0.5, 1.0], [1.0, -1.0], [-1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    counts = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        ],
        dtype=torch.float32,
    )
    split = ResponseSplit(
        cone_response=cones,
        spike_counts=counts,
        valid_mask=torch.ones_like(counts, dtype=torch.bool),
        source_ids=("source-a", "source-b"),
        context_ids=("low", "high"),
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
        time_axis_seconds=np.arange(6, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(2, dtype=np.float32),
        normalization_std=np.ones(2, dtype=np.float32),
        fingerprint="stage05-fixture",
        input_identity=synthetic_input_identity(2, split.source_ids),
    )


def _altered_split(split: ResponseSplit) -> ResponseSplit:
    return replace(
        split,
        cone_response=-split.cone_response,
        spike_counts=1.0 - split.spike_counts,
    )
