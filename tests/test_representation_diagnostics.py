from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation import decoder_diagnostics
from evaluation.decoder_diagnostics import (
    decoder_coverage,
    decode_with_fit,
    fit_global_decoder,
)
from evaluation.reconstruction import (
    causal_ema,
    fit_causal_ema_alpha,
    reconstruction_metrics,
)
from evaluation.representation_diagnostics import (
    DecoderExamples,
    representation_diagnostics,
)
from models.decoder.local_decoder import TiedLocalDecoder
from training.augmentation import AugmentedClip
from training.config import load_config
from training.data import PreparedClip
from training.schedule import objective_weights
from training.validation_clips import fixed_validation_clips


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


def test_regularized_probe_uses_calibrated_gain_as_underdetermined_prior() -> None:
    rates = torch.zeros(2, 5, 2, 3)
    weights = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    )
    prior_gain = torch.tensor([[0.4, 1.2, 2.0], [1.6, 0.7, 0.3]])
    target = torch.linspace(-0.5, 0.5, 20).reshape(2, 5, 2)

    fit = decoder_diagnostics.fit_regularized_tied_decoder_probe(
        rates,
        target,
        weights,
        gain_max=5.0,
        prior_gain=prior_gain,
    )

    assert torch.allclose(fit.unit_gain, prior_gain, atol=1e-6)
    assert fit.ridge_strength > 0.0
    assert fit.gain_clipped_fraction == 0.0


def test_decoder_accepts_effective_calibration_values() -> None:
    decoder = TiedLocalDecoder(unit_count=3, cone_count=2, gain_max=5.0)
    gain = torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    bias = torch.tensor([0.1, -0.2])

    decoder.initialize(gain, bias)

    assert torch.allclose(decoder.unit_gain, gain)
    assert torch.allclose(decoder.cone_bias, bias)


def test_balanced_validation_has_opposing_scenarios_per_source() -> None:
    config = load_config(ROOT / "configs" / "cone_reconstruction_diagnostic.yaml")
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


def test_objective_schedule_keeps_repulsion_out_of_bootstrap() -> None:
    config = load_config(ROOT / "configs" / "cone_reconstruction_diagnostic.yaml")
    start = objective_weights(0, config)
    bootstrap_end = objective_weights(
        config.training.reconstruction_bootstrap_steps,
        config,
    )
    ramp_end = objective_weights(config.training.budget_ramp_end_step, config)

    assert start.phenotype_repulsion == 0.0
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


def test_representation_diagnostics_distinguishes_current_and_fixed_readouts() -> None:
    rates = torch.zeros(1, 4, 2, 2)
    rates[0, :, 0] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    rates[0, :, 1] = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 2.0]]
    )
    weights = torch.eye(2)
    target_gain = torch.tensor([[1.5, 1.5], [0.5, 0.5]])
    target_bias = torch.tensor([0.2, -0.3])
    target = decode_with_fit(rates, weights, target_gain, target_bias)
    examples = DecoderExamples(
        rates=rates,
        generator_potential=rates,
        target=target,
        noisy_input=target,
        source_ids=("source",),
    )
    current_decoder = TiedLocalDecoder(unit_count=2, cone_count=2, gain_max=5.0)
    fixed_decoder = TiedLocalDecoder(unit_count=2, cone_count=2, gain_max=5.0)
    current_decoder.initialize(target_gain, target_bias)
    fixed_decoder.initialize(torch.full_like(target_gain, 0.1), torch.zeros(2))

    diagnostics = representation_diagnostics(
        current_decoder,
        fixed_decoder,
        examples,
        examples,
        weights,
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.zeros(2),
        0.25,
    )

    assert diagnostics.current_decoder.mse == pytest.approx(0.0, abs=1e-10)
    assert diagnostics.fixed_calibrated_decoder.mse > 0.0
    assert diagnostics.posthoc_tied_decoder_probe.mse < 1e-4
    assert diagnostics.posthoc_tied_decoder_probe_source_cv_mse >= 0.0
    assert diagnostics.source_metrics[0].source_id == "source"
