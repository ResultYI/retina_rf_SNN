from __future__ import annotations

import inspect

import torch

from evaluation.mechanistic_retina.h1_teachers import (
    H1TeacherRequest,
    build_matched_h1_teachers,
)
from evaluation.mechanistic_retina.spike_banks import (
    generate_nested_spike_bank,
    slice_spike_bank,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.sampled import (
    SampledTrainingRequest,
    train_sampled_model,
)


def _model():
    torch.manual_seed(7)
    return build_mechanistic_retina(
        MechanisticRetinaConfig(lag_steps=4),
        torch.tensor(
            [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.04, 0.00], [0.11, 0.00]], dtype=torch.float32),
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def test_bernoulli_master_bank_is_deterministic_and_nested() -> None:
    probability = torch.linspace(0.05, 0.60, 48).reshape(2, 12, 2)

    first = generate_nested_spike_bank(probability, probability.flip(0), seed=31001, max_trials=8)
    second = generate_nested_spike_bank(probability, probability.flip(0), seed=31001, max_trials=8)
    t2 = slice_spike_bank(first, 2)
    t8 = slice_spike_bank(first, 8)

    assert torch.equal(first.train_spikes, second.train_spikes)
    assert torch.equal(t2.train_spikes, t8.train_spikes[:, :2])
    assert first.train_sha256 == second.train_sha256
    assert first.train_sha256 != first.validation_sha256
    assert t2.train_spikes.shape == (2, 2, 12, 2)


def test_sampled_training_uses_fixed_steps_and_no_rf_target() -> None:
    model = _model()
    cones = torch.randn(3, 12, 4)
    probability = torch.full((3, 12, 2), 0.2)
    bank = slice_spike_bank(
        generate_nested_spike_bank(probability, probability, seed=11, max_trials=2),
        2,
    )
    mask = torch.ones_like(bank.train_spikes, dtype=torch.bool)

    result = train_sampled_model(
        SampledTrainingRequest(
            model=model,
            train_cones=cones,
            train_spikes=bank.train_spikes,
            train_mask=mask,
            validation_cones=cones,
            validation_spikes=bank.validation_spikes,
            validation_mask=mask,
            validation_probability=probability,
            steps=2,
            checkpoint_steps=(0, 1, 2),
            learning_rate=0.03,
            batch_size=2,
            seed=19,
        )
    )

    assert tuple(point.step for point in result.checkpoints) == (0, 1, 2)
    assert result.gradients_finite
    assert tuple(inspect.signature(expected_bernoulli_nll).parameters) == (
        "logits",
        "teacher_probability",
        "valid_mask",
    )


def test_h1_teachers_share_base_calibration_and_match_mean_rate() -> None:
    model = _model()
    cones = torch.randn(3, 12, 4)
    validation = torch.randn(2, 12, 4)
    rf = torch.randn(2, 4, 4) * 0.1
    train_mask = torch.ones(3, 12, 2, dtype=torch.bool)
    validation_mask = torch.ones(2, 12, 2, dtype=torch.bool)

    teachers = build_matched_h1_teachers(
        H1TeacherRequest(
            model=model,
            train_cones=cones,
            validation_cones=validation,
            base_rf=rf,
            train_mask=train_mask,
            validation_mask=validation_mask,
            response_bias=-2.0,
        )
    )

    assert torch.equal(teachers.absent_bias, torch.full((2,), -2.0))
    assert abs(teachers.present_mean_rate - teachers.absent_mean_rate) <= 1e-7
    assert not torch.equal(teachers.present_train_probability, teachers.absent_train_probability)
    assert torch.isfinite(teachers.teacher_h1_component_rf).all()


def test_h1_clamp_changes_only_h1_surround_input() -> None:
    model = _model()
    cones = torch.randn(2, 12, 4)
    history = torch.zeros(2, 12, 2)

    full = model.forward_sequence(cones, observed_counts=history)
    clamped = model.forward_sequence(
        cones,
        observed_counts=history,
        clamps=frozenset({PathwayClamp.H1}),
    )

    assert torch.count_nonzero(full.h1_surround_contribution) > 0
    assert torch.count_nonzero(clamped.h1_surround_contribution) == 0
    assert torch.isfinite(full.logits - clamped.logits).all()
