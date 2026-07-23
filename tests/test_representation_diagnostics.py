from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.decoder_diagnostics import (
    decoder_coverage,
    decode_with_fit,
    fit_global_decoder,
    fit_tied_decoder_ceiling,
)
from evaluation.reconstruction import (
    causal_ema,
    fit_causal_ema_alpha,
    reconstruction_metrics,
)
from models.decoder.local_decoder import TiedLocalDecoder
from training.augmentation import AugmentedClip, fixed_validation_clips
from training.config import load_config
from training.data import PreparedClip
from training.schedule import objective_weights


ROOT = Path(__file__).resolve().parents[1]


def test_global_decoder_calibration_recovers_scale_and_bias() -> None:
    rates = torch.zeros(1, 4, 2, 2)
    rates[0, :, 0] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    rates[0, :, 1] = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 2.0]]
    )
    weights = torch.eye(2)
    expected_gain = torch.tensor([[1.5, 1.5], [0.5, 0.5]])
    expected_bias = torch.tensor([0.2, -0.3])
    target = decode_with_fit(
        rates,
        weights,
        expected_gain,
        expected_bias,
    )

    fit = fit_global_decoder(rates, target, weights, gain_max=5.0)

    assert torch.allclose(fit.unit_gain, expected_gain, atol=1e-5)
    assert torch.allclose(fit.cone_bias, expected_bias, atol=1e-5)
    assert fit.train_mse == pytest.approx(0.0, abs=1e-10)


def test_tied_decoder_ceiling_is_never_worse_than_global_fit() -> None:
    generator = torch.Generator().manual_seed(7)
    rates = torch.rand(2, 5, 2, 3, generator=generator)
    weights = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    )
    target_gain = torch.tensor([[0.4, 1.2, 2.0], [1.6, 0.7, 0.3]])
    target_bias = torch.tensor([0.1, -0.2])
    target = decode_with_fit(rates, weights, target_gain, target_bias)

    global_fit = fit_global_decoder(rates, target, weights, gain_max=5.0)
    ceiling = fit_tied_decoder_ceiling(rates, target, weights, gain_max=5.0)

    assert ceiling.train_mse <= global_fit.train_mse + 1e-10
    assert ceiling.train_mse == pytest.approx(0.0, abs=1e-9)


def test_decoder_accepts_effective_calibration_values() -> None:
    decoder = TiedLocalDecoder(unit_count=3, cone_count=2, gain_max=5.0)
    gain = torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    bias = torch.tensor([0.1, -0.2])

    decoder.initialize(gain, bias)

    assert torch.allclose(decoder.unit_gain, gain)
    assert torch.allclose(decoder.cone_bias, bias)


def test_balanced_validation_has_opposing_scenarios_per_source() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    clip = PreparedClip(
        clean=torch.ones(config.data.sequence_steps, 2),
        source_id="source",
    )

    clips = fixed_validation_clips(
        [clip],
        config.data,
        seed=31,
        device=torch.device("cpu"),
    )

    assert len(clips) == 2
    assert {row.metadata["scenario"] for row in clips} == {
        "low_gain_high_noise_to_high_gain_low_noise",
        "high_gain_low_noise_to_low_gain_high_noise",
    }
    assert clips[0].metadata["transition_step"] == clips[1].metadata["transition_step"]


def test_causal_ema_alpha_is_fit_on_training_observations() -> None:
    clean = torch.tensor([[[0.0], [1.0], [1.0], [1.0]]])
    noisy = torch.tensor([[[0.0], [2.0], [0.0], [2.0]]])

    alpha = fit_causal_ema_alpha(noisy, clean, candidates=(0.0, 0.5, 0.9))
    filtered = causal_ema(noisy, alpha)
    metrics = reconstruction_metrics(
        prediction=filtered,
        target=clean,
        train_mean=torch.tensor([0.0]),
        noisy_input=noisy,
        ema_alpha=alpha,
    )

    assert alpha == 0.5
    assert metrics.causal_ema_mse < metrics.noisy_current_mse


def test_objective_schedule_separates_repulsion_and_constraints() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    start = objective_weights(0, config)
    bootstrap_end = objective_weights(
        config.training.reconstruction_bootstrap_steps,
        config,
    )
    ramp_end = objective_weights(config.training.budget_ramp_end_step, config)

    assert start.phenotype_repulsion == config.objective.phenotype_repulsion_weight
    assert start.wiring == 0.0
    assert bootstrap_end.phenotype_repulsion == 0.0
    assert bootstrap_end.wiring == 0.0
    assert ramp_end.wiring == config.objective.wiring_weight
    assert start.variance == ramp_end.variance == config.objective.variance_weight
    assert start.homeostasis == ramp_end.homeostasis == config.objective.homeostasis_weight


def test_decoder_coverage_reports_edge_imbalance() -> None:
    weights = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
    )
    positions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])

    coverage = decoder_coverage(weights, positions)

    assert coverage.minimum == pytest.approx(0.5)
    assert coverage.maximum == pytest.approx(1.5)
    assert coverage.edge_to_center_ratio < 1.0


def test_augmented_clip_batches_are_concatenated_once() -> None:
    first = AugmentedClip(
        noisy_input=torch.zeros(1, 3, 2),
        clean_target=torch.ones(1, 3, 2),
        metadata={"source_id": "a"},
    )
    second = AugmentedClip(
        noisy_input=torch.full((1, 3, 2), 2.0),
        clean_target=torch.full((1, 3, 2), 3.0),
        metadata={"source_id": "b"},
    )

    batch = AugmentedClip.stack((first, second))

    assert batch.noisy_input.shape == (2, 3, 2)
    assert batch.clean_target.shape == (2, 3, 2)
