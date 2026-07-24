from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import models.decoder.local_decoder as decoder_module
import training.runtime as runtime
from loss.retina import RetinaObjective
from training.bootstrap import calibrate_view_consistency_weight
from training.config import load_config
from training.schedule import objective_weights
from training.trainer import RetinaTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_cross_fitted_readout_backpropagates_to_each_held_out_source() -> None:
    # Given: three source rows and tied decoder geometry.
    generator = torch.Generator().manual_seed(7)
    readout = torch.randn(3, 5, 2, 2, generator=generator, requires_grad=True)
    target = torch.randn(3, 5, 2, generator=generator)
    geometry = decoder_module.TiedReadoutGeometry(
        spatial_weights=torch.eye(2),
        prior_gain=torch.full((2, 2), 0.5),
        gain_max=5.0,
    )

    # When: each row is decoded by a fit learned from the other source rows.
    result = decoder_module.cross_fitted_tied_reconstruction(
        readout,
        target,
        geometry,
    )
    result.loss.backward()

    # Then: held-out predictions retain a gradient to every representation row.
    assert readout.grad is not None
    assert torch.all(readout.grad.flatten(1).norm(dim=1) > 0)
    assert result.fold_mse.shape == (3,)


def test_training_source_sampling_is_without_replacement() -> None:
    # Given: more training sources than one optimizer batch.
    generator = torch.Generator().manual_seed(19)

    # When: one batch of source indices is sampled.
    indices = runtime.sample_unique_source_indices(12, 4, generator)

    # Then: the batch contains four distinct sources.
    assert indices.shape == (4,)
    assert torch.unique(indices).numel() == 4


def test_view_consistency_anneals_before_diagnostic_end() -> None:
    # Given: a 50-step representation diagnostic.
    config = load_config(ROOT / "configs" / "cone_reconstruction_diagnostic.yaml")
    config = replace(
        config,
        training=replace(config.training, decoder_freeze_steps=50),
    )

    # When: objective weights are evaluated across the diagnostic.
    start = objective_weights(0, config)
    decay_midpoint = objective_weights(35, config)
    auxiliary_off = objective_weights(40, config)

    # Then: generator deep supervision is gone for the final ten steps.
    assert start.view_consistency_scale == 1.0
    assert decay_midpoint.view_consistency_scale == pytest.approx(0.5)
    assert auxiliary_off.view_consistency_scale == 0.0


def test_view_consistency_weight_matches_core_gradient_ratio() -> None:
    # Given: rate and generator losses with a known four-to-one gradient ratio.
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    rate_loss = (2.0 * parameter).square()
    generator_loss = parameter.square()

    # When: the generator auxiliary is calibrated once.
    weight = calibrate_view_consistency_weight(
        rate_loss,
        generator_loss,
        (parameter,),
    )

    # Then: its weighted gradient is one quarter of the rate gradient.
    assert weight == pytest.approx(1.0)


def test_bootstrap_calibration_state_round_trips_in_checkpoint() -> None:
    # Given: a trainer with calibrated view consistency.
    config = load_config(ROOT / "configs" / "cone_reconstruction_diagnostic.yaml")
    model = torch.nn.Linear(1, 1)
    decoder = torch.nn.Linear(1, 1)
    objective = RetinaObjective(
        rho_energy=1.0,
        variance_floor=0.01,
        phenotype_temperature=1.0,
        homeostasis_rate_min=0.001,
    )
    trainer = RetinaTrainer(model, decoder, objective, config, 1.0)
    trainer.bootstrap_state.view_consistency_base_weight = 2.5
    trainer.bootstrap_state.view_consistency_calibrated_step = 0
    trainer.bootstrap_state.initial_generator_variance_reference = torch.ones(
        2,
        1,
    )
    sampling = torch.Generator().manual_seed(1)
    augmentation = torch.Generator().manual_seed(2)

    # When: the checkpoint is restored into an equivalent trainer.
    payload = trainer.checkpoint_payload(sampling, augmentation)
    restored = RetinaTrainer(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        objective,
        config,
        1.0,
    )
    restored.restore(
        payload,
        torch.Generator(),
        torch.Generator(),
    )

    # Then: one-time calibration is not repeated after resume.
    assert restored.bootstrap_state.view_consistency_base_weight == 2.5
    assert restored.bootstrap_state.view_consistency_calibrated_step == 0
    assert torch.equal(
        restored.bootstrap_state.initial_generator_variance_reference,
        torch.ones(2, 1),
    )
