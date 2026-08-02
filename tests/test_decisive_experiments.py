from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.point_process_teacher import generate_teacher_responses
from data.rgc_response_export import write_rgc_response
from data.synthetic_teacher import fit_teacher_input_normalization
from evaluation.factorial_contrasts import factorial_contrasts
from evaluation.factorial_reachability import (
    FactorialJacobianInput,
    summarize_factorial_jacobian,
)
from evaluation.target_gradient_comparison import summarize_target_gradients
from evaluation.teacher_identifiability import reconstruct_teacher_targets
from evaluation.trial_power import TrialPowerRequest, audit_trial_power_curve
from evaluation.type_gain_reachability import summarize_type_gain_jacobian
from scripts import audit_response_gradients as gradient_audit
from scripts import run_decisive_experiments as decisive
from tests.decisive_variant_fixtures import (
    FakeResponseTrainer,
    FakeVariantModel,
    ParameterSharingMode,
    StopAfterModelConstruction,
    prepared_response_data,
    response_config,
)


def test_decisive_cli_smoke_bounds_defaults_but_preserves_explicit_values() -> None:
    # Given
    argv = [
        "--config",
        "config.yaml",
        "--checkpoint",
        "checkpoint.pt",
        "--strong-validation",
        "strong.h5",
        "--output",
        "out.json",
        "--smoke",
        "--power-seeds",
        "7",
    ]

    # When
    args = decisive.parse_decisive_args(argv)

    # Then
    assert args.power_seeds == 7
    assert args.bootstrap_iterations == 20
    assert args.power_bootstrap_iterations == 20
    assert args.oracle_iterations == 1


@pytest.mark.parametrize("mode", ("type_blind", "cell_only", "shuffled_type"))
def test_decisive_reconstruction_preserves_variant_identity(
    monkeypatch: pytest.MonkeyPatch,
    mode: ParameterSharingMode,
) -> None:
    # Given
    config = response_config(mode, seed=37)
    data = prepared_response_data()
    build_model = Mock(return_value=FakeVariantModel())
    monkeypatch.setattr(decisive, "load_response_config", Mock(return_value=config))
    monkeypatch.setattr(decisive, "prepare_response_data", Mock(return_value=data))
    monkeypatch.setattr(decisive, "validate_experiment_input", Mock())
    monkeypatch.setattr(decisive, "load_type_priors", Mock(return_value=object()))
    monkeypatch.setattr(decisive, "build_response_retina_model", build_model)
    monkeypatch.setattr(decisive, "ResponseTrainer", FakeResponseTrainer)
    monkeypatch.setattr(
        decisive,
        "load_response_checkpoint",
        Mock(side_effect=StopAfterModelConstruction),
    )

    # When
    with pytest.raises(StopAfterModelConstruction):
        decisive._run(
            argparse.Namespace(
                config="config.yaml",
                checkpoint="checkpoint.pt",
                strong_validation="strong.h5",
                output="out.json",
                device="cpu",
                probe_steps=1,
                bootstrap_iterations=1,
                power_seeds=1,
                power_bootstrap_iterations=1,
                oracle_iterations=1,
                smoke=True,
            )
        )

    # Then
    assert build_model.call_args.kwargs["parameter_sharing_mode"] == mode
    assert build_model.call_args.kwargs["parameter_sharing_seed"] == 37


@pytest.mark.parametrize("mode", ("type_blind", "cell_only", "shuffled_type"))
def test_gradient_audit_reconstruction_preserves_variant_identity(
    monkeypatch: pytest.MonkeyPatch,
    mode: ParameterSharingMode,
) -> None:
    # Given
    config = response_config(mode, seed=41)
    data = prepared_response_data()
    build_model = Mock(return_value=FakeVariantModel())
    monkeypatch.setattr(gradient_audit, "load_response_config", Mock(return_value=config))
    monkeypatch.setattr(gradient_audit, "prepare_response_data", Mock(return_value=data))
    monkeypatch.setattr(gradient_audit, "validate_experiment_input", Mock())
    monkeypatch.setattr(gradient_audit, "load_type_priors", Mock(return_value=object()))
    monkeypatch.setattr(gradient_audit, "build_response_retina_model", build_model)
    monkeypatch.setattr(gradient_audit, "ResponseTrainer", FakeResponseTrainer)
    monkeypatch.setattr(
        gradient_audit,
        "load_response_checkpoint",
        Mock(side_effect=StopAfterModelConstruction),
    )
    monkeypatch.setattr(
        gradient_audit.sys,
        "argv",
        [
            "audit_response_gradients.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "out.json",
        ],
    )

    # When
    with pytest.raises(StopAfterModelConstruction):
        gradient_audit.main()

    # Then
    assert build_model.call_args.kwargs["parameter_sharing_mode"] == mode
    assert build_model.call_args.kwargs["parameter_sharing_seed"] == 41


def test_reconstructed_teacher_targets_match_generated_probabilities(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    normalization = fit_teacher_input_normalization(cones)
    generated = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        np.arange(80) * 0.005,
        trials=4,
        seed=11,
        adaptive=True,
        teacher_normalization=normalization,
    )
    path = tmp_path / "teacher.h5"
    write_rgc_response(
        path,
        generated.session,
        teacher_kernels=generated.kernels,
        teacher_normalization=normalization,
    )

    reconstructed = reconstruct_teacher_targets(path)

    assert np.allclose(
        reconstructed.expected_probabilities,
        generated.expected_probabilities,
        atol=1e-7,
    )
    assert np.allclose(
        reconstructed.conditional_probabilities,
        generated.conditional_probabilities,
        atol=1e-7,
    )


def test_target_gradient_summary_keeps_type_specific_components() -> None:
    named_parameters = (
        ("rgc.a.type_base_raw", nn.Parameter(torch.zeros(2))),
        ("h1.raw_gain", nn.Parameter(torch.zeros(1))),
    )
    gradients = (torch.tensor([1.0, -1.0]), torch.tensor([2.0]))

    summary = summarize_target_gradients(
        torch.tensor(0.5),
        named_parameters,
        gradients,
        ("midget", "parasol"),
    )

    assert np.isclose(summary.raw_gradient_norm, np.sqrt(6.0))
    assert summary.type_vectors[0].values == (1.0,)
    assert summary.type_vectors[1].values == (-1.0,)
    assert summary.type_differential.opposition_cosine > 0.99


def test_type_gain_jacobian_summary_detects_independent_control() -> None:
    jacobian = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    summary = summarize_type_gain_jacobian(
        jacobian,
        current=np.asarray([0.0, 0.0]),
        target=np.asarray([0.5, -0.5]),
        lower_delta=np.asarray([-1.0, -1.0]),
        upper_delta=np.asarray([1.0, 1.0]),
    )

    assert summary.rank == 2
    assert summary.condition_number == 2.0
    assert abs(summary.sensitivity_cosine) < 1e-12
    assert summary.bounded_least_squares_residual < 1e-10


def test_factorial_contrasts_separate_type_and_polarity() -> None:
    # Given
    gains = np.asarray([0.85, 0.90, 1.10, 1.15])

    # When
    contrasts = factorial_contrasts(gains)

    # Then
    assert np.isclose(contrasts.common, 1.0)
    assert np.isclose(contrasts.type, 0.125)
    assert np.isclose(contrasts.polarity, 0.025)
    assert np.isclose(contrasts.interaction, 0.0)


def test_factorial_jacobian_summary_predicts_all_contrasts() -> None:
    # Given
    jacobian = np.eye(4, dtype=np.float64)
    current = np.zeros(4, dtype=np.float64)
    target = np.asarray([-0.02, -0.01, 0.01, 0.02], dtype=np.float64)

    # When
    summary = summarize_factorial_jacobian(
        FactorialJacobianInput(
            jacobian,
            current,
            target,
            np.full(4, -1.0),
            np.full(4, 1.0),
        )
    )

    # Then
    assert summary.rank == 4
    assert summary.bounded_least_squares_residual < 1e-10
    assert np.allclose(summary.predicted_cell_gains, target)
    assert np.isclose(
        summary.predicted_contrasts.type,
        factorial_contrasts(target).type,
    )


def test_trial_power_curve_reports_each_requested_trial_count(tmp_path: Path) -> None:
    # Given
    rng = np.random.default_rng(17)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    normalization = fit_teacher_input_normalization(cones)
    generated = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        np.arange(80) * 0.005,
        trials=2,
        seed=19,
        adaptive=True,
        teacher_normalization=normalization,
    )
    path = tmp_path / "power.h5"
    write_rgc_response(
        path,
        generated.session,
        teacher_kernels=generated.kernels,
        teacher_normalization=normalization,
    )

    # When
    curve = audit_trial_power_curve(
        TrialPowerRequest(
            path,
            trial_counts=(2, 4),
            monte_carlo_seeds=2,
            bootstrap_iterations=20,
            probe_steps=32,
            seed=23,
        )
    )

    # Then
    assert tuple(point.trial_count for point in curve.points) == (2, 4)
    assert all(0.0 <= point.direction_recovery_rate <= 1.0 for point in curve.points)
    assert all(0.0 <= point.ci_support_rate <= 1.0 for point in curve.points)
