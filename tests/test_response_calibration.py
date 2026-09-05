from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.rgc_response import ResponseTargetKind
from evaluation.response_calibration import (
    CalibrationError,
    LogitCalibrationRequest,
    ThresholdCalibrationRequest,
    fit_logit_calibration,
    fit_threshold_calibration,
)
from evaluation.response_predictions import (
    ResponsePredictionRequest,
    ResponsePredictionTensors,
    collect_response_predictions,
)
from models.response_snn import build_response_retina_model
from tests.calibration_fixture import (
    calibration_config,
    calibration_data,
    calibration_trainer,
)
from training.response_data import PreparedResponseData, ResponseSplit
from training.response_trainer import ResponseTrainer


def test_positive_affine_calibration_extracts_frozen_logit_signal() -> None:
    # Given
    logits = torch.tensor([[[[-0.1], [-0.05], [0.05], [0.1]]]])
    targets = torch.tensor([[[[0.0], [0.0], [1.0], [1.0]]]])
    predictions = ResponsePredictionTensors(
        logits,
        logits,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
    )
    baseline_rates = torch.tensor([0.5])

    # When
    intercept = fit_logit_calibration(
        LogitCalibrationRequest(
            predictions,
            predictions,
            ResponseTargetKind.BERNOULLI,
            baseline_rates,
            "intercept",
            40,
        )
    )
    affine = fit_logit_calibration(
        LogitCalibrationRequest(
            predictions,
            predictions,
            ResponseTargetKind.BERNOULLI,
            baseline_rates,
            "affine",
            40,
        )
    )

    # Then
    assert affine.scales[0] > 0
    assert affine.validation_metrics.nll < intercept.validation_metrics.nll
    assert torch.equal(predictions.logits, logits)


def test_threshold_recomputation_matches_observed_history_forward() -> None:
    # Given
    trainer = calibration_trainer()
    request = ResponsePredictionRequest(
        trainer.model,
        trainer.data.train,
        trainer.config.training.burn_in_steps,
        trainer.device,
    )
    before = collect_response_predictions(request)
    with torch.no_grad():
        trainer.model.rgc.threshold.type_base_raw.add_(0.1)

    # When
    after = collect_response_predictions(request)
    recomputed = trainer.model.rgc.logits_from_generator(before.generator_potential)

    # Then
    assert torch.allclose(after.generator_potential, before.generator_potential)
    assert torch.allclose(after.logits, recomputed)


def test_threshold_calibration_changes_only_threshold_parameters() -> None:
    # Given
    trainer = calibration_trainer()
    train = collect_response_predictions(
        ResponsePredictionRequest(
            trainer.model,
            trainer.data.train,
            trainer.config.training.burn_in_steps,
            trainer.device,
        )
    )
    original = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    # When
    result = fit_threshold_calibration(
        ThresholdCalibrationRequest(
            trainer.model,
            train,
            train,
            trainer.data.target_kind,
            trainer.baseline_rates,
            20,
            0.001,
        )
    )

    # Then
    assert result.changed_parameter_names
    assert all(name.startswith("rgc.threshold.") for name in result.changed_parameter_names)
    assert all(
        torch.equal(value, dict(trainer.model.named_parameters())[name])
        for name, value in original.items()
    )


def test_logit_calibration_rejects_non_bernoulli_targets() -> None:
    # Given
    values = torch.zeros(1, 1, 2, 1)
    predictions = ResponsePredictionTensors(
        values,
        values,
        values,
        torch.ones_like(values, dtype=torch.bool),
    )

    # When
    request = LogitCalibrationRequest(
        predictions,
        predictions,
        ResponseTargetKind.POISSON,
        torch.ones(1),
        "intercept",
        1,
    )

    # Then
    with pytest.raises(CalibrationError, match="Bernoulli"):
        fit_logit_calibration(request)


def test_logit_calibration_rejects_misaligned_tensors() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 1)
    predictions = ResponsePredictionTensors(
        logits,
        logits,
        torch.zeros(1, 1, 1, 1),
        torch.ones_like(logits, dtype=torch.bool),
    )

    # When
    request = LogitCalibrationRequest(
        predictions,
        predictions,
        ResponseTargetKind.BERNOULLI,
        torch.ones(1),
        "intercept",
        1,
    )

    # Then
    with pytest.raises(CalibrationError, match="share a shape"):
        fit_logit_calibration(request)


def test_stage0_calibration_uses_train_split_only_and_changes_only_bias() -> None:
    # Given
    train_data = calibration_data()
    heldout_changed = _with_heldout_spikes(train_data, value=1.0)

    # When
    trainer, initial = _response_bias_trainer(train_data, stage0_enabled=True)
    changed_trainer, changed_initial = _response_bias_trainer(
        heldout_changed,
        stage0_enabled=True,
    )

    # Then
    assert trainer.stage0_result is not None
    assert changed_trainer.stage0_result is not None
    assert trainer.stage0_result.post_train_nll < trainer.stage0_result.pre_train_nll
    assert torch.allclose(
        trainer.model.rgc.response_bias,
        changed_trainer.model.rgc.response_bias,
    )
    assert initial.keys() == changed_initial.keys()
    for name, value in initial.items():
        current = dict(trainer.model.named_parameters())[name]
        if name == "rgc.response_bias":
            assert not torch.equal(current, value)
        else:
            assert torch.equal(current, value)
    assert trainer.optimizer_step == 0


def test_disabled_stage0_calibration_is_exact_noop() -> None:
    # Given / When
    trainer, initial = _response_bias_trainer(
        calibration_data(),
        stage0_enabled=False,
    )

    # Then
    assert trainer.stage0_result is None
    assert trainer.optimizer_step == 0
    for name, value in initial.items():
        assert torch.equal(dict(trainer.model.named_parameters())[name], value)


def test_threshold_freeze_excludes_threshold_from_optimizer_and_step_delta() -> None:
    # Given
    trainer, _ = _response_bias_trainer(
        calibration_data(),
        stage0_enabled=False,
        freeze_threshold=True,
    )
    threshold_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if name.startswith("rgc.threshold.")
    }
    group_names = _optimizer_group_names(trainer)

    # When
    trainer.train_step(
        trainer.data.train.cone_response[:1],
        trainer.data.train.spike_counts[0, :1],
        trainer.data.train.valid_mask[0, :1],
    )

    # Then
    assert threshold_before
    assert all(
        not parameter.requires_grad
        for name, parameter in trainer.model.named_parameters()
        if name.startswith("rgc.threshold.")
    )
    assert all(
        name not in group_names
        for name in threshold_before
    )
    for name, value in threshold_before.items():
        assert torch.equal(dict(trainer.model.named_parameters())[name], value)


def test_optimizer_groups_are_named_stable_and_cellwise() -> None:
    # Given / When
    trainer, _ = _response_bias_trainer(
        calibration_data(),
        stage0_enabled=False,
        freeze_threshold=True,
    )
    groups = trainer.optimizer.param_groups
    group_names = [_group_name(group) for group in groups]
    membership = _optimizer_group_names(trainer)

    # Then
    assert group_names == ["response_bias", "rgc", "upstream"]
    assert [group["lr"] for group in groups] == [
        trainer.config.training.response_bias_lr,
        trainer.config.training.rgc_lr,
        trainer.config.training.learning_rate,
    ]
    assert membership["response_bias"] == ("rgc.response_bias",)
    assert "rgc.synaptic_gain_raw" in membership["rgc"]
    assert all(
        not name.startswith("rgc.threshold.")
        for names in membership.values()
        for name in names
    )
    assert any(name.startswith("h1.") for name in membership["upstream"])
    assert any(name.startswith("bipolar.") for name in membership["upstream"])
    assert any(name.startswith("amacrine.") for name in membership["upstream"])


def _response_bias_trainer(
    data: PreparedResponseData,
    *,
    stage0_enabled: bool,
    freeze_threshold: bool = False,
) -> tuple[ResponseTrainer, dict[str, torch.Tensor]]:
    config = calibration_config()
    config = replace(
        config,
        model=replace(
            config.model,
            enable_response_bias=True,
            enable_synaptic_gain=True,
        ),
        training=replace(
            config.training,
            stage0_calibration_enabled=stage0_enabled,
            freeze_threshold=freeze_threshold,
        ),
    )
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
        enable_response_bias=True,
        enable_synaptic_gain=True,
        synaptic_gain_min=0.1,
        synaptic_gain_max=4.0,
        synaptic_gain_init=1.0,
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return ResponseTrainer(model, config, data, torch.device("cpu")), initial


def _optimizer_group_names(
    trainer: ResponseTrainer,
) -> dict[str, tuple[str, ...]]:
    names_by_id = {
        id(parameter): name for name, parameter in trainer.model.named_parameters()
    }
    return {
        _group_name(group): tuple(
            names_by_id[id(parameter)] for parameter in group["params"]
        )
        for group in trainer.optimizer.param_groups
    }


def _group_name(group) -> str:
    name = group["name"]
    assert isinstance(name, str)
    return name


def _with_heldout_spikes(
    data: PreparedResponseData,
    *,
    value: float,
) -> PreparedResponseData:
    validation = _filled_split(data.validation, value)
    test = _filled_split(data.test, value)
    return replace(data, validation=validation, test=test)


def _filled_split(split: ResponseSplit, value: float) -> ResponseSplit:
    return ResponseSplit(
        cone_response=split.cone_response,
        spike_counts=torch.full_like(split.spike_counts, value),
        valid_mask=split.valid_mask,
        source_ids=split.source_ids,
        context_ids=split.context_ids,
    )
