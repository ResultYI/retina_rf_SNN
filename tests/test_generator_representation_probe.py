from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.decoder_diagnostics import decoder_coverage, decode_with_fit
from evaluation.representation_diagnostics import (
    DecoderExamples,
    RepresentationDiagnostics,
    SourceDecoderMetrics,
    compare_representation_diagnostics,
    representation_diagnostics,
)
from evaluation.reconstruction import reconstruction_metrics
from models.decoder.local_decoder import TiedLocalDecoder
from training.experiment_cli import parse_experiment_args


def test_generator_probe_detects_information_missing_from_filtered_rates() -> None:
    rates = torch.zeros(2, 4, 2, 2)
    generator = torch.zeros_like(rates)
    generator[:, :, 0] = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]]
    )
    generator[:, :, 1] = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 2.0]]]
    )
    weights = torch.eye(2)
    gain = torch.tensor([[1.5, 1.5], [0.5, 0.5]])
    bias = torch.tensor([0.2, -0.3])
    target = decode_with_fit(generator, weights, gain, bias)
    examples = DecoderExamples(
        rates=rates,
        target=target,
        noisy_input=target,
        source_ids=("a", "b"),
        generator_potential=generator,
    )
    decoder = TiedLocalDecoder(unit_count=2, cone_count=2, gain_max=5.0)
    decoder.initialize(gain, bias)

    diagnostics = representation_diagnostics(
        decoder,
        decoder,
        examples,
        examples,
        weights,
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.zeros(2),
        0.25,
    )

    assert diagnostics.posthoc_generator_probe.mse < 1e-4
    assert diagnostics.posthoc_tied_decoder_probe.mse > 0.1
    assert diagnostics.posthoc_generator_probe_source_cv_mse < 1e-4
    assert diagnostics.source_metrics[0].posthoc_generator_probe_mse < 1e-4


def test_read_only_representation_resume_mode_is_accepted() -> None:
    arguments = parse_experiment_args(
        [
            "--representation-diagnostic-steps",
            "50",
            "--diagnostics-only",
            "--resume",
            str(Path("checkpoint.pt")),
        ]
    )

    assert arguments.diagnostics_only
    assert arguments.resume == Path("checkpoint.pt")


def test_training_resume_remains_incompatible_with_representation_mode() -> None:
    with pytest.raises(SystemExit):
        parse_experiment_args(
            [
                "--representation-diagnostic-steps",
                "50",
                "--resume",
                str(Path("checkpoint.pt")),
            ]
        )


def test_representation_comparison_includes_generator_probe_delta() -> None:
    reconstruction = reconstruction_metrics(
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1),
        torch.ones(1, 1, 1),
        0.25,
    )
    initial = RepresentationDiagnostics(
        current_decoder=reconstruction,
        fixed_calibrated_decoder=reconstruction,
        posthoc_tied_decoder_probe=reconstruction,
        posthoc_generator_probe=reconstruction,
        posthoc_tied_decoder_probe_train_mse=1.0,
        posthoc_tied_decoder_probe_source_cv_mse=1.0,
        posthoc_generator_probe_train_mse=1.0,
        posthoc_generator_probe_source_cv_mse=1.0,
        ema_alpha=0.25,
        coverage=decoder_coverage(torch.ones(1, 1), torch.zeros(1, 2)),
        source_metrics=(SourceDecoderMetrics("a", 1.0, 1.0, 1.0, 1.0),),
    )
    improved = reconstruction_metrics(
        torch.full((1, 1, 1), 0.5),
        torch.zeros(1, 1, 1),
        torch.zeros(1),
        torch.ones(1, 1, 1),
        0.25,
    )
    selected = RepresentationDiagnostics(
        current_decoder=improved,
        fixed_calibrated_decoder=improved,
        posthoc_tied_decoder_probe=improved,
        posthoc_generator_probe=improved,
        posthoc_tied_decoder_probe_train_mse=0.5,
        posthoc_tied_decoder_probe_source_cv_mse=0.5,
        posthoc_generator_probe_train_mse=0.5,
        posthoc_generator_probe_source_cv_mse=0.5,
        ema_alpha=0.25,
        coverage=initial.coverage,
        source_metrics=(SourceDecoderMetrics("a", 0.5, 0.5, 0.5, 0.5),),
    )

    comparison = compare_representation_diagnostics(initial, selected)

    assert comparison.generator_probe_mse_delta == pytest.approx(-0.75)
    assert comparison.improved_generator_source_count == 1
